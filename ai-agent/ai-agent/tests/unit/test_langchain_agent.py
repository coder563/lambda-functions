from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage, ToolMessage

# ---------------------------------------------------------------------------
# Test 1 — agent returns a plain text answer (no tool use)
# ---------------------------------------------------------------------------

def test_langchain_agent_returns_answer():
    """Agent returns a string answer when no tools are needed."""
    with patch("hello_world.agent_langchain.create_react_agent") as mock_create:
        mock_agent = MagicMock()
        mock_agent.invoke.return_value = {
            "messages": [AIMessage(content="Hello, I am Claude.")]
        }
        mock_create.return_value = mock_agent

        from hello_world.agent_langchain import run_langchain_agent
        result = run_langchain_agent([("user", "Say hello")])

        assert result == "Hello, I am Claude."
        mock_agent.invoke.assert_called_once()


# ---------------------------------------------------------------------------
# Test 2 — agent invokes a tool and returns the final answer
# ---------------------------------------------------------------------------

def test_langchain_agent_calls_tool():
    """Agent calls a tool mid-conversation and returns final answer."""
    with patch("hello_world.agent_langchain.create_react_agent") as mock_create:
        mock_agent = MagicMock()
        # LangGraph returns all messages including intermediate tool messages.
        # The last AIMessage is the final answer.
        mock_agent.invoke.return_value = {
            "messages": [
                AIMessage(content="", tool_calls=[{"name": "get_weather", "args": {"location": "London"}, "id": "1"}]),
                ToolMessage(content="Sunny and 22°C", tool_call_id="1"),
                AIMessage(content="The weather in London is sunny and 22°C.")
            ]
        }
        mock_create.return_value = mock_agent

        from hello_world.agent_langchain import run_langchain_agent
        result = run_langchain_agent([("user", "What is the weather in London?")])

        assert result == "The weather in London is sunny and 22°C."
        mock_agent.invoke.assert_called_once()
