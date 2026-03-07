import json
import os
import boto3


def lambda_handler(event, context):
    session_id = event.get("session_id")
    message = event.get("message") or event.get("prompt")

    # Load existing history from DynamoDB if a session is provided
    history = []
    dynamodb_table = os.environ.get("DYNAMODB_TABLE")
    table = None
    if session_id and dynamodb_table:
        dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
        table = dynamodb.Table(dynamodb_table)
        result = table.get_item(Key={"session_id": session_id})
        if "Item" in result:
            history = result["Item"].get("history", [])

    messages = history + [{"role": "user", "content": message}]

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    response = client.invoke_model(
        modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": messages
        })
    )
    response_body = json.loads(response["body"].read())
    answer = response_body["content"][0]["text"]

    # Save updated history to DynamoDB
    if session_id and dynamodb_table and table is not None:
        table.put_item(Item={
            "session_id": session_id,
            "history": messages + [{"role": "assistant", "content": answer}]
        })

    return {
        "statusCode": 200,
        "body": answer
    }
