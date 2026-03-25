import os
from langchain_aws import ChatBedrock
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from typing import Annotated
from typing_extensions import TypedDict

from hello_world.app import get_weather, calculator, get_current_time, load_history, save_history
from langchain_core.tools import tool as lc_tool


@lc_tool
def get_weather_tool(location: str) -> str:
    """Get the current weather for a given location."""
    return get_weather(location)


@lc_tool
def calculator_tool(expression: str) -> str:
    """Evaluate a mathematical expression and return the result."""
    return calculator(expression)


@lc_tool
def get_current_time_tool() -> str:
    """Get the current date and time."""
    return get_current_time()


TOOLS = [get_weather_tool, calculator_tool, get_current_time_tool]
TOOL_MAP = {t.name: t for t in TOOLS}


class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


def build_graph(checkpointer=None):
    llm = ChatBedrock(
        model_id="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        region_name="us-east-1"
    ).bind_tools(TOOLS)

    def call_llm(state: AgentState):
        response = llm.invoke(state["messages"])
        return {"messages": [response]}

    def execute_tools(state: AgentState):
        last = state["messages"][-1]
        results = []
        for tc in last.tool_calls:
            result = TOOL_MAP[tc["name"]].invoke(tc["args"])
            results.append(ToolMessage(content=result, tool_call_id=tc["id"]))
        return {"messages": results}

    def should_continue(state: AgentState):
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tools"
        return "end"

    graph = StateGraph(AgentState)
    graph.add_node("call_llm", call_llm)
    graph.add_node("execute_tools", execute_tools)

    graph.add_edge(START, "call_llm")
    graph.add_conditional_edges("call_llm", should_continue, {"tools": "execute_tools", "end": END})
    graph.add_edge("execute_tools", "call_llm")

    return graph.compile(checkpointer=checkpointer)


# Module-level graph + checkpointer so state survives warm Lambda invocations.
# Swap MemorySaver for a DynamoDB/Redis-backed checkpointer in production.
_checkpointer = MemorySaver()
_checkpointed_graph = build_graph(checkpointer=_checkpointer)


def graph_checkpointed_handler(event, context):
    """Handler that lets LangGraph manage conversation state via a checkpointer."""
    thread_id = event.get("session_id") or "default"
    message = event.get("message") or event.get("prompt")

    config = {"configurable": {"thread_id": thread_id}}
    result = _checkpointed_graph.invoke(
        {"messages": [HumanMessage(content=message)]},
        config=config,
    )

    last = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
    return {"statusCode": 200, "body": last.content}


def graph_handler(event, context):
    session_id = event.get("session_id")
    message = event.get("message") or event.get("prompt")

    history, table = load_history(session_id, os.environ.get("DYNAMODB_TABLE"))
    lc_history = [HumanMessage(content=m["content"]) if m["role"] == "user" else AIMessage(content=m["content"]) for m in history]
    messages = lc_history + [HumanMessage(content=message)]

    graph = build_graph()
    result = graph.invoke({"messages": messages})

    last = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
    answer = last.content

    if table is not None:
        save_history(table, session_id, history + [{"role": "user", "content": message}], answer)

    return {"statusCode": 200, "body": answer}
