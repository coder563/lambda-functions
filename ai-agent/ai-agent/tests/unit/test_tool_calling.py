import json
import pytest
from unittest.mock import MagicMock, patch, call


def make_bedrock_response(body_dict):
    return {"body": MagicMock(read=lambda: json.dumps(body_dict).encode())}


# What Claude returns when it wants to call a tool
TOOL_USE_RESPONSE = make_bedrock_response({
    "stop_reason": "tool_use",
    "content": [
        {"type": "text", "text": "Let me check the weather for you."},
        {"type": "tool_use", "id": "toolu_123", "name": "get_weather", "input": {"location": "London"}}
    ]
})

# What Claude returns after receiving the tool result
FINAL_RESPONSE = make_bedrock_response({
    "stop_reason": "end_turn",
    "content": [
        {"type": "text", "text": "The weather in London is sunny and 22°C."}
    ]
})


def test_claude_is_called_twice_when_tool_is_used():
    """When Claude requests a tool, the handler should call Bedrock a second time with the tool result."""
    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = [TOOL_USE_RESPONSE, FINAL_RESPONSE]
        mock_client.return_value = mock_bedrock

        from hello_world.app import lambda_handler
        event = {"prompt": "What is the weather in London?"}
        lambda_handler(event, None)

        assert mock_bedrock.invoke_model.call_count == 2


def test_tool_result_is_included_in_second_call():
    """The second Bedrock call should include the tool result in the messages."""
    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = [TOOL_USE_RESPONSE, FINAL_RESPONSE]
        mock_client.return_value = mock_bedrock

        from hello_world.app import lambda_handler
        event = {"prompt": "What is the weather in London?"}
        lambda_handler(event, None)

        second_call_body = json.loads(mock_bedrock.invoke_model.call_args_list[1].kwargs["body"])
        messages = second_call_body["messages"]

        # Last message should be the tool result from the user
        last_message = messages[-1]
        assert last_message["role"] == "user"
        assert last_message["content"][0]["type"] == "tool_result"
        assert last_message["content"][0]["tool_use_id"] == "toolu_123"


def test_final_answer_comes_from_second_call():
    """The handler should return the text from the second Bedrock call."""
    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = [TOOL_USE_RESPONSE, FINAL_RESPONSE]
        mock_client.return_value = mock_bedrock

        from hello_world.app import lambda_handler
        event = {"prompt": "What is the weather in London?"}
        response = lambda_handler(event, None)

        assert response["body"] == "The weather in London is sunny and 22°C."
