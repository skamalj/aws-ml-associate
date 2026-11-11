# Coverage map: every MLA-C02 skill → notebook

Source: [MLA-C02 exam guide](https://docs.aws.amazon.com/aws-certification/latest/machine-learning-engineer-associate-02/machine-learning-engineer-associate-02.html) (4 domains, 12 tasks, 107 skills).

- ✅ runnable hands-on code
- 🔒 code is written but off by default (costs money or needs a manual prerequisite such as a human workforce or Docker)
- 📝 intentionally notes / decision table only

Every notebook section heading carries its skill ID, so you can search a notebook for e.g. `1.2.7`.

## Domain 1: Data Preparation for ML and AI (28%)

| Skill | Topic | Notebook | Status | How it is covered |
|---|---|---|---|---|
| 1.1.1 | Extract data from sources | 1.1 | ✅ | S3, DynamoDB and Athena run; RDS, EFS, EBS and OpenSearch as snippets and notes |
| 1.1.2 | Storage decisions | 1.1 | ✅ | S3 storage classes and a lifecycle rule; "which store when" table |
| 1.1.3 | Troubleshoot ingestion capacity | 1.1 | 📝 | Throttling, shard limits, small-file problem; one small Kinesis demo |
| 1.1.4 | Streaming ingestion | 1.1 | ✅ | Kinesis Data Streams and Firehose to S3; MSK and Flink as notes |
| 1.1.5 | Data formats | 1.1 | ✅ | CSV vs JSON vs Parquet vs ORC, size and read-time comparison |
| 1.1.6 | Merge multiple sources | 1.1 | ✅ | pandas join, Glue crawler with Athena join, Glue Spark script |
| 1.1.7 | Vector databases | 1.1 | ✅ | S3 Vectors index runs; OpenSearch Serverless 🔒; pgvector SQL shown |
| 1.1.8 | Diverse data types | 00, 1.1 | ✅ | Text, image and audio files stored with metadata |
| 1.1.9 | Ingest into Feature Store | 1.1 | ✅ | Create a feature group and ingest |
| 1.2.1 | Transform with AWS tools | 1.2a | ✅ | Glue job, DataBrew recipe, SageMaker Processing; EMR and Data Wrangler as notes |
| 1.2.2 | Create and manage features | 1.2a | ✅ | Online and offline store reads, point-in-time query |
| 1.2.3 | Transform streaming data | 1.2a | ✅ | Lambda transform on telematics; Spark streaming as notes |
| 1.2.4 | Feature engineering | 1.2a | ✅ | Scaling, standardization, binning, log, splitting, encoding |
| 1.2.5 | Embedding models | 1.2b | ✅ | Titan text and multimodal embeddings; dimensions |
| 1.2.6 | Text pre-processing | 1.2b | ✅ | Tokenization and token counts; domain augmentation with an FM |
| 1.2.7 | RAG document preparation | 1.2b | ✅ | Fixed, semantic and hierarchical chunking; metadata files |
| 1.2.8 | Mask, redact, anonymize | 1.2b | ✅ | Comprehend PII detection and redaction, hashing; Macie as a note |
| 1.2.9 | Data for fine-tuning, continued pre-training, distillation | 1.2b | ✅ | Build and validate the three JSONL formats |
| 1.3.1 | Validate data quality | 1.3 | ✅ | Glue Data Quality rules; DataBrew profile job |
| 1.3.2 | Label and annotate | 1.3 | 🔒 | Ground Truth job code and manifest (needs a human workforce) |
| 1.3.3 | Bias in data | 1.3 | ✅ | Clarify pre-training bias; stratified split, shuffle |
| 1.3.4 | Multimodal bias metrics | 1.3 | ✅ | Bias metrics on tabular data, adjuster notes and image labels |
| 1.3.5 | Class imbalance | 1.3 | ✅ | SMOTE, class weights; text and image augmentation |
| 1.3.6 | AI training data integrity | 1.3 | ✅ | Prompt-response pair validation; safety screening with a Guardrail |
| 1.3.7 | Clean data | 1.3 | ✅ | Outliers, imputation, dedup on `claims_v2` |

## Domain 2: ML Model and Foundation Model Development (24%)

| Skill | Topic | Notebook | Status | How it is covered |
|---|---|---|---|---|
| 2.1.1 | Select Bedrock FMs | 2.1 | ✅ | List models; same prompt across 3 models |
| 2.1.2 | Fine-tuning strategies | 2.1 | 📝 | Decision table (prompting, RAG, fine-tuning, continued pre-training, distillation); hands-on is 2.2.8 |
| 2.1.3 | Compare models and algorithms | 2.1 | 📝 | Built-in algorithm cheat sheet; JumpStart listing |
| 2.1.4 | Custom vs managed vs FM | 2.1 | 📝 | Decision table and scenarios |
| 2.1.5 | RAG architecture patterns | 2.1 | 📝 | Pattern table; hands-on is 3.1.8 and 3.2.8 |
| 2.1.6 | ML performance, time, cost | 2.2a | ✅ | Two training configs compared on time, cost and AUC |
| 2.1.7 | AI performance, latency, cost | 2.1 | ✅ | Latency and token cost measured per model |
| 2.1.8 | AWS AI services | 2.1 | ✅ | Textract, Rekognition, Comprehend, Transcribe on claim files |
| 2.2.1 | Built-in algorithms | 2.2a | ✅ | XGBoost fraud model |
| 2.2.2 | Script mode | 2.2a | ✅ | scikit-learn entry script |
| 2.2.3 | Hyperparameter tuning | 2.2a | ✅ | Automatic model tuning, Bayesian, warm start |
| 2.2.4 | Reduce training time | 2.2a | ✅ | Early stopping, 2-instance distributed, Spot with checkpoints |
| 2.2.5 | Overfitting, underfitting, forgetting | 2.2a, 2.2b | ✅ | Regularization and curves run; catastrophic forgetting as notes |
| 2.2.6 | Combine models | 2.2a | ✅ | Ensemble; cheap-model-first cascade |
| 2.2.7 | Basic hyperparameters | 2.2a | ✅ | Epochs, batch size, learning-rate effect |
| 2.2.8 | Prompt engineering, fine-tuning | 2.2b | ✅ | Converse API, few-shot, JSON output, tool use; fine-tuning job 🔒 |
| 2.2.9 | Optimize retrieval and embeddings | 2.2b | ✅ | Chunk size and embedding dimension vs recall@k |
| 2.3.1 | Reproducible experiments | 2.3a, 2.3b | ✅ | MLflow on SageMaker; Bedrock evaluation job; prompt versions |
| 2.3.2 | Baselines, drift detection | 2.3a | ✅ | Baseline created here; drift run in 4.1a |
| 2.3.3 | Shadow vs production variant | 4.1a | ✅ | Run in 4.1a, where the endpoints exist; notes in 2.3a |
| 2.3.4 | Explain model outputs | 2.3a | ✅ | Clarify SHAP on flagged claims |
| 2.3.5 | Debug convergence | 2.3a | ✅ | Bad learning-rate demo; Debugger rules |
| 2.3.6 | Evaluation techniques | 2.3a, 2.3b | ✅ | Classification and regression metrics; GenAI metrics |
| 2.3.7 | Human evaluation | 2.3b | 🔒 | A2I human loop and Bedrock human evaluation code (needs a workforce) |
| 2.3.8 | NLP metrics | 2.3b | ✅ | BLEU, ROUGE, BERTScore, cosine similarity |
| 2.3.9 | AI evaluation | 2.3b | ✅ | LLM-as-a-judge, quality and bias checks |
| 2.3.10 | RAG monitoring | 2.3b, 3.2b | ✅ | Recall@k and MRR locally; KB evaluation job in 3.2b |

## Domain 3: Deployment and Orchestration of ML and AI Workflows (24%)

| Skill | Topic | Notebook | Status | How it is covered |
|---|---|---|---|---|
| 3.1.1 | Compute and deployment targets | 3.1a | ✅ | Decision table; deployments below |
| 3.1.2 | Orchestrators, multi-model, multi-container | 3.1a | ✅ | Multi-model endpoint, inference pipeline; orchestrator table |
| 3.1.3 | Inference strategies | 3.1a | ✅ | Real-time, serverless, async, batch transform |
| 3.1.4 | FM deployment options | 3.1a | ✅ | On-demand, cross-region profile; provisioned and JumpStart 🔒 |
| 3.1.5 | Models built outside AWS | 3.1a | ✅ | Locally trained model to SageMaker; Custom Model Import 🔒 |
| 3.1.6 | Agents, tools, protocols | 3.1b | ✅ | Bedrock Agent with a Lambda tool; Strands agent; MCP via Gateway; A2A; multi-agent |
| 3.1.7 | FM hosting configuration | 3.1a | ✅ | Inference components, resource allocation |
| 3.1.8 | RAG configurations | 3.2b | ✅ | Hybrid search, metadata filter, reranking |
| 3.2.1 | On-demand vs provisioned | 3.2a | ✅ | Serverless provisioned concurrency; Bedrock Provisioned Throughput 🔒 |
| 3.2.2 | Automate provisioning | 3.2a | ✅ | CloudFormation stack with cross-stack outputs |
| 3.2.3 | Build containers | 3.2a | 🔒 | Dockerfile, ECR push (needs Docker) |
| 3.2.4 | Endpoints in a VPC | 3.2a | ✅ | VpcConfig, VPC endpoints |
| 3.2.5 | Deploy programmatically | 3.2a | ✅ | Same deploy via SDK, boto3 and CLI |
| 3.2.6 | Auto scaling metrics | 3.2a | ✅ | Target tracking on invocations per instance; step scaling |
| 3.2.7 | Bedrock Knowledge Bases | 3.2b | ✅ | KB on S3 Vectors, data source, sync |
| 3.2.8 | Retrieval pipelines | 3.2b | ✅ | Retrieve, RetrieveAndGenerate, custom pipeline |
| 3.2.9 | Agent state management | 3.2b | ✅ | Session attributes; AgentCore Memory |
| 3.2.10 | GPU scaling | 3.2a | 🔒 | GPU metrics, scale-to-zero config |
| 3.2.11 | Agentic infrastructure | 3.1b | ✅ | Deploy to AgentCore Runtime |
| 3.3.1 | Deployment strategies, rollback | 3.3a | ✅ | Canary with alarm auto-rollback |
| 3.3.2 | Code* services | 3.3a | ✅ | CodeBuild and CodePipeline created; CodeCommit, CodeDeploy and CodeConnections as notes; troubleshooting table |
| 3.3.3 | Training and inference jobs | 3.3a | ✅ | Pipeline steps |
| 3.3.4 | Automated tests in CI/CD | 3.3a | ✅ | Unit test, model-quality gate, endpoint smoke test |
| 3.3.5 | Retraining mechanisms | 3.3a | ✅ | EventBridge triggers the pipeline |
| 3.3.6 | Model versions | 3.3a | ✅ | Model Registry approval; MLflow |
| 3.3.7 | Manage prompts | 3.3b | ✅ | Prompt Management versions and variants |
| 3.3.8 | Agent pipelines and versions | 3.3b | ✅ | Agent versions and aliases, runtime versions |
| 3.3.9 | Prompt and model testing | 3.3b | ✅ | Prompt regression test suite |
| 3.3.10 | Fine-tuned model versioning | 3.3b | 🔒 | Code depends on the 🔒 fine-tuning job |
| 3.3.11 | RAG and KB refresh | 3.3b | ✅ | S3 event, Lambda, ingestion job |

## Domain 4: Operating, Monitoring, and Securing ML and AI Solutions (24%)

| Skill | Topic | Notebook | Status | How it is covered |
|---|---|---|---|---|
| 4.1.1 | Production monitoring | 4.1a, 4.1b | ✅ | Drift pipeline; CloudWatch GenAI observability |
| 4.1.2 | Workflow anomalies | 4.1a | ✅ | Alarms on endpoint errors and failed jobs |
| 4.1.3 | Data distribution change | 4.1a | ✅ | Model Monitor against `claims_v2` |
| 4.1.4 | A/B testing | 4.1a | ✅ | Two production variants, weights shifted |
| 4.1.5 | Agent performance | 4.1b | ✅ | Tool failures, truncated output, coordination failures |
| 4.1.6 | FM monitoring | 4.1b | ✅ | Invocation logging, evaluation runs |
| 4.2.1 | Instance families | 4.2 | 📝 | Table; Inference Recommender code 🔒 |
| 4.2.2 | Troubleshooting tools | 4.2, 4.1b | ✅ | Logs Insights, X-Ray, AgentCore Observability |
| 4.2.3 | Dashboards | 4.2 | ✅ | CloudWatch dashboard for both systems |
| 4.2.4 | Capacity optimization | 4.2 | 📝 | Notes; scaling hands-on is 3.2.6 |
| 4.2.5 | Cost tools, quotas | 4.2 | ✅ | Budgets, Cost Explorer by tag, Service Quotas |
| 4.2.6 | Purchasing options | 4.2 | 📝 | Savings Plans vs Spot vs on-demand table |
| 4.2.7 | FM inference cost | 4.2 | ✅ | Cost-per-question calculator |
| 4.2.8 | Agent resource use | 4.2 | ✅ | Tokens and steps per agent session |
| 4.2.9 | FM usage optimization | 4.2 | ✅ | Prompt caching, batch inference, prompt routing |
| 4.2.10 | AI cost patterns | 4.2 | ✅ | Token, embedding and vector storage cost math |
| 4.3.1 | Secure CI/CD | 4.3a | ✅ | ECR scan and Inspector findings; CodeGuru as a note |
| 4.3.2 | Least privilege | 4.3a | ✅ | Scoped policies on bucket, model, KB |
| 4.3.3 | IAM policies and roles | 4.3a | ✅ | Data scientist vs app roles |
| 4.3.4 | Audit and compliance | 4.3a | ✅ | CloudTrail lookup, Config rule |
| 4.3.5 | Debug security issues | 4.3a | ✅ | Trigger and diagnose AccessDenied; policy simulator |
| 4.3.6 | VPC isolation | 4.3a | ✅ | VPC, subnets, security group and endpoints via CloudFormation |
| 4.3.7 | Security risks | 4.3a, 4.3b | ✅ | Prompt-injection demo; ML risks as notes |
| 4.3.8 | Credential types for FMs | 4.3b | ✅ | IAM vs Bedrock API key |
| 4.3.9 | Guardrails | 4.3b | ✅ | PII, denied topics, grounding check, ApplyGuardrail |

## Totals

| | ✅ | 🔒 | 📝 | Total |
|---|---|---|---|---|
| Domain 1 | 23 | 1 | 1 | 25 |
| Domain 2 | 22 | 1 | 4 | 27 |
| Domain 3 | 27 | 3 | 0 | 30 |
| Domain 4 | 22 | 0 | 3 | 25 |
| **All** | **94** | **5** | **8** | **107** |
