import json
import pytest
from unittest.mock import MagicMock, patch


def make_bedrock_response(body_dict):
    return {"body": MagicMock(read=lambda: json.dumps(body_dict).encode())}


def make_tool_use_response(tool_name, tool_id, tool_input):
    return make_bedrock_response({
        "stop_reason": "tool_use",
        "content": [
            {"type": "tool_use", "id": tool_id, "name": tool_name, "input": tool_input}
        ]
    })


def test_calculator_tool_is_called_and_result_fed_back():
    """When Claude requests calculator, agent should execute it and feed result back."""
    FINAL_RESPONSE = make_bedrock_response({
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "The answer is 4."}]
    })

    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = [
            make_tool_use_response("calculator", "toolu_001", {"expression": "2 + 2"}),
            FINAL_RESPONSE
        ]
        mock_client.return_value = mock_bedrock

        from hello_world.app import lambda_handler
        response = lambda_handler({"prompt": "What is 2 + 2?"}, None)

        assert mock_bedrock.invoke_model.call_count == 2

        second_call_body = json.loads(mock_bedrock.invoke_model.call_args_list[1].kwargs["body"])
        last_message = second_call_body["messages"][-1]
        assert last_message["role"] == "user"
        assert last_message["content"][0]["type"] == "tool_result"
        assert last_message["content"][0]["content"] == "4"
        assert response["body"] == "The answer is 4."


def test_get_current_time_tool_is_called_and_result_fed_back():
    """When Claude requests get_current_time, agent should execute it and feed result back."""
    FINAL_RESPONSE = make_bedrock_response({
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "The current time has been retrieved."}]
    })

    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = [
            make_tool_use_response("get_current_time", "toolu_002", {}),
            FINAL_RESPONSE
        ]
        mock_client.return_value = mock_bedrock

        from hello_world.app import lambda_handler
        response = lambda_handler({"prompt": "What time is it?"}, None)

        assert mock_bedrock.invoke_model.call_count == 2

        second_call_body = json.loads(mock_bedrock.invoke_model.call_args_list[1].kwargs["body"])
        last_message = second_call_body["messages"][-1]
        assert last_message["role"] == "user"
        assert last_message["content"][0]["type"] == "tool_result"
        assert len(last_message["content"][0]["content"]) > 0
        assert response["statusCode"] == 200


def test_get_weather_tool_is_called_and_result_fed_back():
    """When Claude requests get_weather, agent should execute it and feed result back."""
    FINAL_RESPONSE = make_bedrock_response({
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "The weather in London is sunny."}]
    })

    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = [
            make_tool_use_response("get_weather", "toolu_003", {"location": "London"}),
            FINAL_RESPONSE
        ]
        mock_client.return_value = mock_bedrock

        from hello_world.app import lambda_handler
        response = lambda_handler({"prompt": "What is the weather in London?"}, None)

        assert mock_bedrock.invoke_model.call_count == 2

        second_call_body = json.loads(mock_bedrock.invoke_model.call_args_list[1].kwargs["body"])
        last_message = second_call_body["messages"][-1]
        assert last_message["role"] == "user"
        assert last_message["content"][0]["type"] == "tool_result"
        assert "London" in last_message["content"][0]["content"]
        assert response["statusCode"] == 200


def test_calculator_and_get_current_time_used_in_sequence():
    """When Claude requests two tools in sequence, both results should be fed back correctly."""
    FINAL_RESPONSE = make_bedrock_response({
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "2+2 is 4 and the current time has been retrieved."}]
    })

    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.side_effect = [
            make_tool_use_response("calculator", "toolu_001", {"expression": "2 + 2"}),
            make_tool_use_response("get_current_time", "toolu_002", {}),
            FINAL_RESPONSE
        ]
        mock_client.return_value = mock_bedrock

        from hello_world.app import lambda_handler
        response = lambda_handler({"prompt": "What is 2+2 and what time is it?"}, None)

        assert mock_bedrock.invoke_model.call_count == 3

        # Second call should contain calculator result
        second_call_body = json.loads(mock_bedrock.invoke_model.call_args_list[1].kwargs["body"])
        second_last_message = second_call_body["messages"][-1]
        assert second_last_message["content"][0]["type"] == "tool_result"
        assert second_last_message["content"][0]["content"] == "4"

        # Third call should contain get_current_time result
        third_call_body = json.loads(mock_bedrock.invoke_model.call_args_list[2].kwargs["body"])
        third_last_message = third_call_body["messages"][-1]
        assert third_last_message["content"][0]["type"] == "tool_result"
        assert len(third_last_message["content"][0]["content"]) > 0

        assert response["statusCode"] == 200
