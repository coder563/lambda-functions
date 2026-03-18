import boto3
import os
import pytest
from moto import mock_aws
from unittest.mock import patch, MagicMock
from langchain_core.messages import AIMessage

os.environ["DYNAMODB_TABLE"] = "test-conversations"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"


def make_mock_agent(answer):
    """Helper: returns a mock agent that always responds with the given answer."""
    mock_agent = MagicMock()
    mock_agent.invoke.return_value = {
        "messages": [AIMessage(content=answer)]
    }
    return mock_agent


# ---------------------------------------------------------------------------
# Test 1 — langchain_handler returns statusCode 200 and the answer
# ---------------------------------------------------------------------------

def test_langchain_handler_returns_answer():
    """langchain_handler returns 200 and the answer string."""
    with patch("hello_world.agent_langchain.create_react_agent") as mock_create:
        mock_create.return_value = make_mock_agent("Hello from LangChain!")

        from hello_world.agent_langchain import langchain_handler
        result = langchain_handler({"message": "Say hello"}, {})

        assert result["statusCode"] == 200
        assert result["body"] == "Hello from LangChain!"


# ---------------------------------------------------------------------------
# Test 2 — history is saved to DynamoDB when session_id is provided
# ---------------------------------------------------------------------------

@mock_aws
def test_langchain_handler_saves_history():
    """After invoking, the session history should be stored in DynamoDB."""
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    dynamodb.create_table(
        TableName="test-conversations",
        KeySchema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "session_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST"
    )

    with patch("hello_world.agent_langchain.create_react_agent") as mock_create:
        mock_create.return_value = make_mock_agent("Hello from LangChain!")

        from hello_world.agent_langchain import langchain_handler
        langchain_handler({"session_id": "lc-session-1", "message": "Hello"}, {})

    table = dynamodb.Table("test-conversations")
    result = table.get_item(Key={"session_id": "lc-session-1"})
    assert "Item" in result
    assert result["Item"]["session_id"] == "lc-session-1"


# ---------------------------------------------------------------------------
# Test 3 — previous history is passed into the agent as context
# ---------------------------------------------------------------------------

@mock_aws
def test_langchain_handler_loads_history():
    """If history exists for a session, it should be included in the agent input."""
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb.create_table(
        TableName="test-conversations",
        KeySchema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "session_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST"
    )
    table.put_item(Item={
        "session_id": "lc-session-2",
        "history": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
    })

    with patch("hello_world.agent_langchain.create_react_agent") as mock_create:
        mock_agent = make_mock_agent("Continuing our chat!")
        mock_create.return_value = mock_agent

        from hello_world.agent_langchain import langchain_handler
        langchain_handler({"session_id": "lc-session-2", "message": "And 3+3?"}, {})

        call_args = mock_agent.invoke.call_args
        messages = call_args[0][0]["messages"]
        # First two messages should be the loaded history
        assert messages[0] == ("user", "What is 2+2?")
        assert messages[1] == ("assistant", "4")
