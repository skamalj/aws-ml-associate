"""Lambda tool for the SafeDrive claims agent: look up a claim in DynamoDB.

Works as a Bedrock Agents action group (function schema) and as a plain Lambda call.
"""
import json
import os

import boto3

table = boto3.resource("dynamodb").Table(os.environ.get("CLAIMS_TABLE", "safedrive-claims"))


def lookup(claim_id):
    item = table.get_item(Key={"claim_id": claim_id}).get("Item")
    if not item:
        return {"found": False, "claim_id": claim_id}
    return {"found": True, **{k: str(v) for k, v in item.items()}}


def handler(event, context):
    if "actionGroup" not in event:                       # direct invocation
        return lookup(event.get("claim_id", ""))
    params = {p["name"]: p["value"] for p in event.get("parameters", [])}
    result = lookup(params.get("claim_id", ""))
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": event["actionGroup"],
            "function": event.get("function", ""),
            "functionResponse": {"responseBody": {"TEXT": {"body": json.dumps(result)}}},
        },
    }
