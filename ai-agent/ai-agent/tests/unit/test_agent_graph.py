import json
import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage


# --- Test 1: simple answer with no tool use ---
def test_graph_agent_returns_answer():
    with patch("hello_world.agent_graph.ChatBedrock") as mock_chat:
        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm

        # LLM responds directly with no tool calls
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="Hello! How can I help?")

        from hello_world.agent_graph import build_graph
        graph = build_graph()

        result = graph.invoke({"messages": [HumanMessage(content="Say hello")]})

        last = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
        assert last.content == "Hello! How can I help?"


# --- Test 2: tool use then final answer ---
def test_graph_agent_calls_tool():
    with patch("hello_world.agent_graph.ChatBedrock") as mock_chat:
        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm

        # First call: Claude requests a tool
        tool_call = {"name": "get_weather_tool", "args": {"location": "London"}, "id": "tc_001", "type": "tool_call"}
        mock_llm.invoke.side_effect = [
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="The weather in London is sunny and 22°C.")
        ]

        from hello_world.agent_graph import build_graph
        graph = build_graph()

        result = graph.invoke({"messages": [HumanMessage(content="Weather in London?")]})

        last = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
        assert "London" in last.content


# --- Test 3: graph_handler with no session_id (no DynamoDB save) ---
def test_graph_handler_no_session():
    with patch("hello_world.agent_graph.ChatBedrock") as mock_chat, \
         patch("hello_world.agent_graph.load_history") as mock_load, \
         patch("hello_world.agent_graph.save_history") as mock_save:

        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="The answer is 42.")
        mock_load.return_value = ([], None)  # no table → no save

        from hello_world.agent_graph import graph_handler
        result = graph_handler({"message": "What is 6 * 7?"}, None)

        assert result["statusCode"] == 200
        assert result["body"] == "The answer is 42."
        mock_save.assert_not_called()


# --- Test 4: graph_handler saves history when session_id is present ---
def test_graph_handler_saves_history():
    with patch("hello_world.agent_graph.ChatBedrock") as mock_chat, \
         patch("hello_world.agent_graph.load_history") as mock_load, \
         patch("hello_world.agent_graph.save_history") as mock_save:

        mock_table = MagicMock()
        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.return_value = AIMessage(content="It's sunny in London.")
        mock_load.return_value = ([], mock_table)

        from hello_world.agent_graph import graph_handler
        result = graph_handler({"session_id": "sess-1", "message": "Weather?"}, None)

        assert result["statusCode"] == 200
        mock_save.assert_called_once()


# --- Test 5: checkpointer preserves state across two calls on the same thread_id ---
def test_checkpointed_graph_same_thread_shares_state():
    with patch("hello_world.agent_graph.ChatBedrock") as mock_chat:
        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = [
            AIMessage(content="Got it, your name is Alice."),
            AIMessage(content="Your name is Alice, you told me earlier."),
        ]

        from hello_world.agent_graph import build_graph
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_graph(checkpointer=MemorySaver())
        config = {"configurable": {"thread_id": "thread-abc"}}

        graph.invoke({"messages": [HumanMessage(content="My name is Alice.")]}, config=config)
        result = graph.invoke({"messages": [HumanMessage(content="What is my name?")]}, config=config)

        last = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
        assert "Alice" in last.content


# --- Test 6: different thread_ids are isolated ---
def test_checkpointed_graph_different_threads_are_isolated():
    with patch("hello_world.agent_graph.ChatBedrock") as mock_chat:
        mock_llm = MagicMock()
        mock_chat.return_value = mock_llm
        mock_llm.bind_tools.return_value = mock_llm
        mock_llm.invoke.side_effect = [
            AIMessage(content="Got it, your name is Alice."),
            AIMessage(content="I don't know your name."),
        ]

        from hello_world.agent_graph import build_graph
        from langgraph.checkpoint.memory import MemorySaver

        graph = build_graph(checkpointer=MemorySaver())

        graph.invoke(
            {"messages": [HumanMessage(content="My name is Alice.")]},
            config={"configurable": {"thread_id": "thread-1"}},
        )
        result = graph.invoke(
            {"messages": [HumanMessage(content="What is my name?")]},
            config={"configurable": {"thread_id": "thread-2"}},
        )

        last = next(m for m in reversed(result["messages"]) if isinstance(m, AIMessage))
        assert "Alice" not in last.content
