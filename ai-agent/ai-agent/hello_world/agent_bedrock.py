import json
import logging

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
