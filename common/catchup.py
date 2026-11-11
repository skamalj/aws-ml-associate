"""Catch-up helpers: build whatever an earlier notebook would have created.

Every ensure_* function is idempotent: it returns the value stored in the shared
state file if it exists, otherwise it creates the resource and saves it.  This lets
you open any notebook directly without running all the earlier ones.
"""
import io
import json
import os
import time
import zipfile
from decimal import Decimal

from botocore.exceptions import ClientError

from . import features as F
from . import utils as U

RAW = f"{U.PREFIX}/raw"
PROCESSED = f"{U.PREFIX}/processed"
CLAIMS_TABLE = "safedrive-claims"
KB_NAME = "safedrive-policy-kb"
VECTOR_INDEX = "policy-docs"
ENDPOINT_NAME = "safedrive-fraud"
AGENT_NAME = "safedrive-claims-agent"
TOOL_LAMBDA = "safedrive-claim-lookup"
GUARDRAIL_NAME = "safedrive-guardrail"


def _exists(fn, **kwargs):
    try:
        fn(**kwargs)
        return True
    except ClientError:
        return False


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------
def ensure_raw_data():
    """Generate the synthetic dataset and upload it to s3://<bucket>/safedrive/raw/."""
    state = U.load_state()
    if state.get("raw_prefix"):
        return state["raw_prefix"]
    from .generate_data import generate
    local = generate(os.path.join(U.REPO_ROOT, "data"))
    n = U.upload_dir(local, RAW)
    print(f"uploaded {n} files to {U.s3_uri(RAW)}")
    U.save_state(raw_prefix=RAW, bucket=U.bucket(), region=U.REGION)
    ensure_audio()
    return RAW


def ensure_audio():
    """Create a short call recording with Amazon Polly (used for Amazon Transcribe)."""
    key = f"{RAW}/audio/call_sample.mp3"
    if U.get_state("audio_key"):
        return key
    text = ("Hello, I want to report a claim. My policy number is P 0 0 0 1 2 3. "
            "Yesterday evening another car hit my rear bumper at a traffic signal. "
            "Nobody was hurt and I have filed a police report.")
    polly = U.client("polly")
    speech = polly.synthesize_speech(Text=text, OutputFormat="mp3", VoiceId="Joanna")
    U.client("s3").put_object(Bucket=U.bucket(), Key=key, Body=speech["AudioStream"].read())
    U.save_state(audio_key=key)
    return key


def ensure_training_data():
    """Create train/validation/test CSVs (label first, no header) for the fraud model."""
    state = U.load_state()
    if state.get("train_s3"):
        return state
    from sklearn.model_selection import train_test_split
    ensure_raw_data()
    feats = F.build_features(U.read_csv(f"{RAW}/claims.csv"), U.read_csv(f"{RAW}/policies.csv"))
    U.write_df(feats, f"{PROCESSED}/features.parquet", fmt="parquet")
    data = feats.drop(columns="claim_id")
    train, rest = train_test_split(data, test_size=0.3, stratify=data[F.LABEL], random_state=42)
    val, test = train_test_split(rest, test_size=0.5, stratify=rest[F.LABEL], random_state=42)
    uris = {}
    for name, part in [("train", train), ("validation", val), ("test", test)]:
        uris[f"{name}_s3"] = U.write_df(part, f"{PROCESSED}/{name}/{name}.csv", header=False)
    # Baseline for Model Monitor / Clarify: features only, with a header row.
    uris["baseline_s3"] = U.write_df(train.drop(columns=F.LABEL), f"{PROCESSED}/baseline/baseline.csv")
    return U.save_state(feature_names=F.FEATURES, features_s3=U.s3_uri(f"{PROCESSED}/features.parquet"), **uris)


def ensure_claims_table(n_items=500):
    """DynamoDB table with a sample of claims (used by the agent's lookup tool)."""
    if U.get_state("claims_table"):
        return CLAIMS_TABLE
    ddb = U.client("dynamodb")
    if not _exists(ddb.describe_table, TableName=CLAIMS_TABLE):
        ddb.create_table(TableName=CLAIMS_TABLE, BillingMode="PAY_PER_REQUEST",
                         AttributeDefinitions=[{"AttributeName": "claim_id", "AttributeType": "S"}],
                         KeySchema=[{"AttributeName": "claim_id", "KeyType": "HASH"}], Tags=U.TAGS)
        ddb.get_waiter("table_exists").wait(TableName=CLAIMS_TABLE)
    ensure_raw_data()
    claims = U.read_csv(f"{RAW}/claims.csv").head(n_items)
    statuses = ["approved", "under_review", "paid", "referred_to_siu"]
    table = U.session.resource("dynamodb").Table(CLAIMS_TABLE)
    with table.batch_writer() as batch:
        for i, row in enumerate(claims.to_dict("records")):
            batch.put_item(Item={
                "claim_id": row["claim_id"], "policy_id": row["policy_id"],
                "claim_type": row["claim_type"], "claim_date": row["claim_date"],
                "claim_amount": Decimal(str(row["claim_amount"])),
                "status": statuses[i % len(statuses)]})
    U.save_state(claims_table=CLAIMS_TABLE)
    return CLAIMS_TABLE


# ---------------------------------------------------------------------------
# Fraud model (traditional ML)
# ---------------------------------------------------------------------------
def ensure_fraud_model():
    """Train the baseline XGBoost fraud model with a SageMaker training job."""
    state = U.load_state()
    if state.get("fraud_model_data"):
        return state["fraud_model_data"]
    import sagemaker
    from sagemaker.estimator import Estimator
    from sagemaker.inputs import TrainingInput
    state = ensure_training_data()
    sm_session = sagemaker.Session(boto_session=U.session, default_bucket=U.bucket())
    image = sagemaker.image_uris.retrieve("xgboost", U.REGION, version="1.7-1")
    est = Estimator(image_uri=image, role=U.role_arn(), instance_count=1, instance_type="ml.m5.large",
                    output_path=U.s3_uri(f"{U.PREFIX}/models"), sagemaker_session=sm_session,
                    base_job_name="safedrive-fraud-xgb", tags=U.TAGS)
    est.set_hyperparameters(objective="binary:logistic", eval_metric="auc", num_round=150,
                            max_depth=5, eta=0.1, subsample=0.8, scale_pos_weight=10)
    est.fit({"train": TrainingInput(state["train_s3"], content_type="text/csv"),
             "validation": TrainingInput(state["validation_s3"], content_type="text/csv")})
    U.save_state(fraud_model_data=est.model_data, fraud_training_job=est.latest_training_job.name,
                 xgb_image=image)
    return est.model_data


def ensure_fraud_endpoint():
    """Deploy the fraud model to a serverless endpoint (no cost while idle)."""
    sm = U.client("sagemaker")
    if _exists(sm.describe_endpoint, EndpointName=ENDPOINT_NAME):
        U.save_state(fraud_endpoint=ENDPOINT_NAME)
        return ENDPOINT_NAME
    model_data = ensure_fraud_model()
    name = f"safedrive-fraud-{int(time.time())}"
    sm.create_model(ModelName=name, ExecutionRoleArn=U.role_arn(), Tags=U.TAGS,
                    PrimaryContainer={"Image": U.get_state("xgb_image"), "ModelDataUrl": model_data})
    sm.create_endpoint_config(
        EndpointConfigName=name, Tags=U.TAGS,
        ProductionVariants=[{"VariantName": "AllTraffic", "ModelName": name,
                             "ServerlessConfig": {"MemorySizeInMB": 2048, "MaxConcurrency": 5}}])
    sm.create_endpoint(EndpointName=ENDPOINT_NAME, EndpointConfigName=name, Tags=U.TAGS)
    sm.get_waiter("endpoint_in_service").wait(EndpointName=ENDPOINT_NAME)
    U.save_state(fraud_endpoint=ENDPOINT_NAME, fraud_model_name=name)
    return ENDPOINT_NAME


# ---------------------------------------------------------------------------
# GenAI: local chunks, Knowledge Base, guardrail, agent
# ---------------------------------------------------------------------------
def ensure_chunks():
    """Chunk the policy documents by heading, embed them, cache on S3. Returns a DataFrame."""
    import pandas as pd
    from . import rag
    key = f"{U.PREFIX}/genai/chunks.parquet"
    if U.get_state("chunks_key"):
        return U.read_parquet(key)
    ensure_raw_data()
    chunks = []
    for k in U.list_keys(f"{RAW}/policy_docs/"):
        if k.endswith(".md"):
            chunks += rag.chunk_by_heading(U.read_bytes(k).decode(), os.path.basename(k))
    df = pd.DataFrame(chunks)
    df["embedding"] = [U.embed(t) for t in df["text"]]
    U.write_df(df, key, fmt="parquet")
    U.save_state(chunks_key=key)
    return df


def ensure_kb():
    """Bedrock Knowledge Base over the policy documents, stored in Amazon S3 Vectors."""
    state = U.load_state()
    if state.get("kb_id"):
        return state["kb_id"]
    ensure_raw_data()
    s3v, agent = U.client("s3vectors"), U.client("bedrock-agent")
    vbucket = f"safedrive-vectors-{U.account_id()}"
    if not _exists(s3v.get_vector_bucket, vectorBucketName=vbucket):
        s3v.create_vector_bucket(vectorBucketName=vbucket)
    if not _exists(s3v.get_index, vectorBucketName=vbucket, indexName=VECTOR_INDEX):
        s3v.create_index(
            vectorBucketName=vbucket, indexName=VECTOR_INDEX, dataType="float32",
            dimension=U.EMBED_DIM, distanceMetric="cosine",
            metadataConfiguration={"nonFilterableMetadataKeys": ["AMAZON_BEDROCK_TEXT", "AMAZON_BEDROCK_METADATA"]})
    index_arn = s3v.get_index(vectorBucketName=vbucket, indexName=VECTOR_INDEX)["index"]["indexArn"]
    kb = agent.create_knowledge_base(
        name=KB_NAME, roleArn=U.role_arn(),
        knowledgeBaseConfiguration={"type": "VECTOR", "vectorKnowledgeBaseConfiguration": {
            "embeddingModelArn": U.model_arn(U.EMBED_MODEL)}},
        storageConfiguration={"type": "S3_VECTORS", "s3VectorsConfiguration": {"indexArn": index_arn}},
        tags={t["Key"]: t["Value"] for t in U.TAGS})["knowledgeBase"]
    kb_id = kb["knowledgeBaseId"]
    U.wait_until(lambda: agent.get_knowledge_base(knowledgeBaseId=kb_id)["knowledgeBase"]["status"],
                 ok=("ACTIVE",), bad=("FAILED",), delay=5, label="knowledge base")
    ds_id = agent.create_data_source(
        knowledgeBaseId=kb_id, name="policy-docs",
        dataSourceConfiguration={"type": "S3", "s3Configuration": {
            "bucketArn": f"arn:aws:s3:::{U.bucket()}", "inclusionPrefixes": [f"{RAW}/policy_docs/"]}},
    )["dataSource"]["dataSourceId"]
    U.save_state(kb_id=kb_id, kb_data_source_id=ds_id, vector_bucket=vbucket, vector_index=VECTOR_INDEX)
    sync_kb(kb_id, ds_id)
    return kb_id


def sync_kb(kb_id=None, ds_id=None):
    """Start a Knowledge Base ingestion job and wait for it to finish."""
    agent = U.client("bedrock-agent")
    kb_id, ds_id = kb_id or U.get_state("kb_id"), ds_id or U.get_state("kb_data_source_id")
    job = agent.start_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id)["ingestionJob"]
    U.wait_until(
        lambda: agent.get_ingestion_job(knowledgeBaseId=kb_id, dataSourceId=ds_id,
                                        ingestionJobId=job["ingestionJobId"])["ingestionJob"]["status"],
        ok=("COMPLETE",), bad=("FAILED", "STOPPED"), delay=10, label="ingestion job")
    return job["ingestionJobId"]


def ensure_guardrail():
    """A basic Bedrock Guardrail (PII masking + one denied topic). Returns (id, version)."""
    state = U.load_state()
    if state.get("guardrail_id"):
        return state["guardrail_id"], state["guardrail_version"]
    bedrock = U.client("bedrock")
    g = bedrock.create_guardrail(
        name=GUARDRAIL_NAME, description="SafeDrive claims assistant guardrail",
        blockedInputMessaging="Sorry, I can only help with SafeDrive insurance questions.",
        blockedOutputsMessaging="Sorry, I cannot share that information.",
        topicPolicyConfig={"topicsConfig": [{
            "name": "legal-advice", "type": "DENY",
            "definition": "Advice on suing, lawsuits or legal strategy against any person or company.",
            "examples": ["Should I sue the other driver?"]}]},
        sensitiveInformationPolicyConfig={"piiEntitiesConfig": [
            {"type": "EMAIL", "action": "ANONYMIZE"}, {"type": "PHONE", "action": "ANONYMIZE"},
            {"type": "US_SOCIAL_SECURITY_NUMBER", "action": "BLOCK"}]},
        tags=[{"key": t["Key"], "value": t["Value"]} for t in U.TAGS])
    version = bedrock.create_guardrail_version(guardrailIdentifier=g["guardrailId"])["version"]
    U.save_state(guardrail_id=g["guardrailId"], guardrail_version=version)
    return g["guardrailId"], version


def ensure_tool_lambda():
    """Deploy the claim-lookup Lambda function used as the agent's tool. Returns its ARN."""
    lam = U.client("lambda")
    ensure_claims_table()
    try:
        return lam.get_function(FunctionName=TOOL_LAMBDA)["Configuration"]["FunctionArn"]
    except ClientError:
        pass
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.write(os.path.join(os.path.dirname(__file__), "lambda_src", "claim_lookup.py"), "claim_lookup.py")
    arn = lam.create_function(
        FunctionName=TOOL_LAMBDA, Runtime="python3.12", Role=U.role_arn(),
        Handler="claim_lookup.handler", Code={"ZipFile": buf.getvalue()}, Timeout=15,
        Environment={"Variables": {"CLAIMS_TABLE": CLAIMS_TABLE}},
        Tags={t["Key"]: t["Value"] for t in U.TAGS})["FunctionArn"]
    lam.get_waiter("function_active_v2").wait(FunctionName=TOOL_LAMBDA)
    lam.add_permission(FunctionName=TOOL_LAMBDA, StatementId="bedrock-agent-invoke",
                       Action="lambda:InvokeFunction", Principal="bedrock.amazonaws.com",
                       SourceAccount=U.account_id())
    U.save_state(tool_lambda_arn=arn)
    return arn


def ensure_agent():
    """Bedrock Agent with the claim-lookup tool. Returns (agent_id, alias_id)."""
    state = U.load_state()
    if state.get("agent_id"):
        return state["agent_id"], state["agent_alias_id"]
    agent = U.client("bedrock-agent")
    lambda_arn = ensure_tool_lambda()
    agent_id = agent.create_agent(
        agentName=AGENT_NAME, agentResourceRoleArn=U.role_arn(), foundationModel=U.TEXT_MODEL,
        idleSessionTTLInSeconds=900,
        instruction=("You are the SafeDrive Insurance claims assistant. Use the claim lookup tool to "
                     "answer questions about a claim. Be brief and never invent claim details."),
        tags={t["Key"]: t["Value"] for t in U.TAGS})["agent"]["agentId"]
    status = lambda: agent.get_agent(agentId=agent_id)["agent"]["agentStatus"]  # noqa: E731
    U.wait_until(status, ok=("NOT_PREPARED", "PREPARED"), bad=("FAILED",), delay=5, label="agent")
    agent.create_agent_action_group(
        agentId=agent_id, agentVersion="DRAFT", actionGroupName="claims-tools",
        actionGroupExecutor={"lambda": lambda_arn},
        functionSchema={"functions": [{
            "name": "get_claim_status", "description": "Look up a SafeDrive claim by its claim ID.",
            "parameters": {"claim_id": {"type": "string", "required": True,
                                        "description": "Claim ID such as C0000001"}}}]})
    agent.prepare_agent(agentId=agent_id)
    U.wait_until(status, ok=("PREPARED",), bad=("FAILED",), delay=5, label="agent")
    alias_id = agent.create_agent_alias(agentId=agent_id, agentAliasName="live")["agentAlias"]["agentAliasId"]
    U.wait_until(lambda: agent.get_agent_alias(agentId=agent_id, agentAliasId=alias_id)["agentAlias"]["agentAliasStatus"],
                 ok=("PREPARED",), bad=("FAILED",), delay=5, label="agent alias")
    U.save_state(agent_id=agent_id, agent_alias_id=alias_id)
    return agent_id, alias_id
