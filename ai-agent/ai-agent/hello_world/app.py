import json
import os
import datetime
import time
import boto3
from botocore.exceptions import ClientError

MAX_ITERATIONS = 10
MAX_RETRIES = 3

TOOLS = [
    {
        "name": "get_weather",
        "description": "Get the current weather for a given location.",
        "input_schema": {
            "type": "object",
            "properties": {
                "location": {
                    "type": "string",
                    "description": "The city to get weather for, e.g. London"
                }
            },
            "required": ["location"]
        }
    },
    {
        "name": "calculator",
        "description": "Evaluate a mathematical expression and return the result.",
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A mathematical expression to evaluate, e.g. '2 + 2' or '10 * 5'"
                }
            },
            "required": ["expression"]
        }
    },
    {
        "name": "get_current_time",
        "description": "Get the current date and time.",
        "input_schema": {
            "type": "object",
            "properties": {}
        }
    }
]


def get_weather(location):
    return f"The weather in {location} is sunny and 22°C."


def calculator(expression):
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


def get_current_time():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def execute_tool(tool_name, tool_input):
    if tool_name == "get_weather":
        return get_weather(tool_input["location"])
    if tool_name == "calculator":
        return calculator(tool_input["expression"])
    if tool_name == "get_current_time":
        return get_current_time()
    raise ValueError(f"Unknown tool: {tool_name}")


def load_history(session_id, dynamodb_table):
    if not session_id or not dynamodb_table:
        return [], None
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb.Table(dynamodb_table)
    result = table.get_item(Key={"session_id": session_id})
    history = result["Item"].get("history", []) if "Item" in result else []
    return history, table


def save_history(table, session_id, messages, answer):
    table.put_item(Item={
        "session_id": session_id,
        "history": messages + [{"role": "assistant", "content": answer}]
    })


def run_agent(messages, client):
    for _ in range(MAX_ITERATIONS):
        for attempt in range(MAX_RETRIES):
            try:
                response = client.invoke_model(
                    modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
                    body=json.dumps({
                        "anthropic_version": "bedrock-2023-05-31",
                        "max_tokens": 512,
                        "tools": TOOLS,
                        "messages": messages
                    })
                )
                break
            except ClientError as e:
                if attempt == MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)
        response_body = json.loads(response["body"].read())

        if response_body.get("stop_reason") == "end_turn":
            answer = next(b["text"] for b in response_body["content"] if b["type"] == "text")
            return answer, messages

        if response_body.get("stop_reason") == "tool_use":
            tool_use_block = next(b for b in response_body["content"] if b["type"] == "tool_use")
            tool_result = execute_tool(tool_use_block["name"], tool_use_block["input"])
            messages = messages + [
                {"role": "assistant", "content": response_body["content"]},
                {"role": "user", "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_block["id"],
                        "content": tool_result
                    }
                ]}
            ]

    return "I wasn't able to complete your request — the agent exceeded the maximum number of steps. Please try rephrasing your question.", messages


def lambda_handler(event, context):
    session_id = event.get("session_id")
    message = event.get("message") or event.get("prompt")

    if not message or not message.strip():
        return {"statusCode": 400, "body": "message is required"}

    history, table = load_history(session_id, os.environ.get("DYNAMODB_TABLE"))
    messages = history + [{"role": "user", "content": message}]

    client = boto3.client("bedrock-runtime", region_name="us-east-1")
    answer, messages = run_agent(messages, client)

    if table is not None:
        save_history(table, session_id, messages, answer)

    return {"statusCode": 200, "body": answer}
