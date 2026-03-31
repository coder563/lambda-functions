import json
import pytest
from unittest.mock import MagicMock, patch


def make_bedrock_response(body_dict):
    return {"body": MagicMock(read=lambda: json.dumps(body_dict).encode())}


TOOL_USE_LONDON = make_bedrock_response({
    "stop_reason": "tool_use",
    "content": [
        {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {"location": "London"}}
    ]
})

TOOL_USE_PARIS = make_bedrock_response({
    "stop_reason": "tool_use",
    "content": [
        {"type": "tool_use", "id": "toolu_2", "name": "get_weather", "input": {"location": "Paris"}}
    ]
})

FINAL_RESPONSE = make_bedrock_response({
    "stop_reason": "end_turn",
    "content": [
        {"type": "text", "text": "London is sunny at 22°C, Paris is cloudy at 18°C."}
    ]
})


def test_agent_loop_handles_multiple_tool_calls():
    """Agent should keep looping until Claude returns end_turn, even across multiple tool calls."""
    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = [TOOL_USE_LONDON, TOOL_USE_PARIS, FINAL_RESPONSE]
        mock_client.return_value = mock_bedrock

        from hello_world.app import lambda_handler
        event = {"prompt": "What is the weather in London and Paris?"}
        response = lambda_handler(event, None)

        assert mock_bedrock.invoke_model.call_count == 3
        assert response["body"] == "London is sunny at 22°C, Paris is cloudy at 18°C."


def test_agent_loop_stops_at_max_iterations():
    """Agent should stop and return an error if the loop exceeds the max iteration limit."""
    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        # Always return tool_use — infinite loop scenario
        mock_bedrock.invoke_model.return_value = make_bedrock_response({
            "stop_reason": "tool_use",
            "content": [
                {"type": "tool_use", "id": "toolu_x", "name": "get_weather", "input": {"location": "London"}}
            ]
        })
        mock_client.return_value = mock_bedrock

        from hello_world.app import lambda_handler
        event = {"prompt": "What is the weather in London?"}
        response = lambda_handler(event, None)

        assert mock_bedrock.invoke_model.call_count <= 10
        assert response["statusCode"] == 200
        assert "maximum number of steps" in response["body"]
