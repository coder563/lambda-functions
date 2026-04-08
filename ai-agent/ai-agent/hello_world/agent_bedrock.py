import json
import logging
import os
import boto3

try:
    from hello_world.app import get_weather, calculator, get_current_time
except ModuleNotFoundError:
    from app import get_weather, calculator, get_current_time

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

TOOL_HANDLERS = {
    "/get-weather": lambda params: get_weather(params["location"]),
    "/calculator": lambda params: calculator(params["expression"]),
    "/get-current-time": lambda _: get_current_time(),
}


def _parse_params(event):
    """Extract parameters from either query params or request body."""
    params = {}
    for p in event.get("parameters") or []:
        params[p["name"]] = p["value"]
    body = event.get("requestBody", {}).get("content", {}).get("application/json", {}).get("properties") or []
    for p in body:
        params[p["name"]] = p["value"]
    return params


def action_group_handler(event, context):
    """
    Handles tool invocations from Amazon Bedrock Agents.
    Bedrock calls this Lambda with a specific action group event format
    and expects a structured response back.
    """
    action_group = event.get("actionGroup")
    api_path = event.get("apiPath")
    http_method = event.get("httpMethod", "GET")

    logger.info(json.dumps({"event": "action_group_call", "actionGroup": action_group, "apiPath": api_path}))

    handler = TOOL_HANDLERS.get(api_path)
    if handler is None:
        return _response(action_group, api_path, http_method, 404, f"Unknown action: {api_path}")

    try:
        params = _parse_params(event)
        result = handler(params)
        return _response(action_group, api_path, http_method, 200, result)
    except Exception as e:
        logger.error(json.dumps({"event": "action_group_error", "apiPath": api_path, "error": str(e)}))
        return _response(action_group, api_path, http_method, 500, f"Error: {e}")


def knowledge_base_handler(event, context):
    """
    Queries a Bedrock Knowledge Base using retrieve_and_generate.
    Bedrock handles the retrieval + feeds context to Claude automatically.
    """
    message = event.get("message") or event.get("prompt")
    if not message or not message.strip():
        return {"statusCode": 400, "body": "message is required"}

    knowledge_base_id = os.environ.get("KNOWLEDGE_BASE_ID")
    if not knowledge_base_id:
        return {"statusCode": 500, "body": "KNOWLEDGE_BASE_ID env var not set"}

    logger.info(json.dumps({"event": "kb_query", "message": message}))

    client = boto3.client("bedrock-agent-runtime", region_name="us-east-1")
    response = client.retrieve_and_generate(
        input={"text": message},
        retrieveAndGenerateConfiguration={
            "type": "KNOWLEDGE_BASE",
            "knowledgeBaseConfiguration": {
                "knowledgeBaseId": knowledge_base_id,
                "modelArn": "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-3-5-sonnet-20241022-v2:0",
            },
        },
    )

    answer = response["output"]["text"]
    logger.info(json.dumps({"event": "kb_query_complete", "answer": answer}))
    return {"statusCode": 200, "body": answer}


def _response(action_group, api_path, http_method, status_code, body):
    return {
        "messageVersion": "1.0",
        "response": {
            "actionGroup": action_group,
            "apiPath": api_path,
            "httpMethod": http_method,
            "httpStatusCode": status_code,
            "responseBody": {
                "application/json": {"body": body}
            },
        },
    }
