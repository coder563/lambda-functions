import os
import datetime
from langchain_aws import ChatBedrock
from langchain_core.tools import tool
from langchain_core.messages import AIMessage
from langgraph.prebuilt import create_react_agent
from hello_world.app import load_history, save_history


@tool
def get_weather(location: str) -> str:
    """Get the current weather for a given location."""
    return f"The weather in {location} is sunny and 22°C."


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TOOLS = [get_weather, calculator, get_current_time]


def run_langchain_agent(messages: list) -> str:
    llm = ChatBedrock(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-east-1"
    )
    agent = create_react_agent(llm, TOOLS)
    result = agent.invoke({"messages": messages})
    last_message = next(
        m for m in reversed(result["messages"])
        if isinstance(m, AIMessage)
    )
    return last_message.content


def langchain_handler(event, context):
    session_id = event.get("session_id")
    message = event.get("message") or event.get("prompt")

    history, table = load_history(session_id, os.environ.get("DYNAMODB_TABLE"))
    messages = [(m["role"], m["content"]) for m in history] + [("user", message)]

    answer = run_langchain_agent(messages)

    if table is not None:
        save_history(table, session_id, history + [{"role": "user", "content": message}], answer)

    return {"statusCode": 200, "body": answer}
