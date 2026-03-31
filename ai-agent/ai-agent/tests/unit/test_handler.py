import json
import pytest
from unittest.mock import MagicMock, patch

from hello_world.app import lambda_handler


def test_handler_returns_200():
    """Lambda should always return a 200 status code"""
    mock_response_body = {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "An AI agent is a system that perceives its environment and takes actions."}]
    }

    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps(mock_response_body).encode())
        }
        mock_client.return_value = mock_bedrock

        event = {"prompt": "What is an AI agent?"}
        response = lambda_handler(event, None)

        assert response["statusCode"] == 200


def test_handler_returns_text_response():
    """Lambda should return Claude's text in the body"""
    expected_text = "An AI agent is a system that perceives its environment and takes actions."
    mock_response_body = {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": expected_text}]
    }

    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps(mock_response_body).encode())
        }
        mock_client.return_value = mock_bedrock

        event = {"prompt": "What is an AI agent?"}
        response = lambda_handler(event, None)

        assert response["body"] == expected_text


def test_handler_uses_default_prompt():
    """Lambda should return 400 if no message is provided"""
    event = {}
    response = lambda_handler(event, None)

    assert response["statusCode"] == 400
    assert "message is required" in response["body"]


def test_handler_sends_prompt_to_bedrock():
    """Lambda should pass the prompt from the event to Bedrock"""
    mock_response_body = {
        "stop_reason": "end_turn",
        "content": [{"type": "text", "text": "Some response"}]
    }

    with patch("hello_world.app.boto3.client") as mock_client:
        mock_bedrock = MagicMock()
        mock_bedrock.invoke_model.return_value = {
            "body": MagicMock(read=lambda: json.dumps(mock_response_body).encode())
        }
        mock_client.return_value = mock_bedrock

        event = {"prompt": "What is an AI agent?"}
        lambda_handler(event, None)

        mock_bedrock.invoke_model.assert_called_once()

        call_args = mock_bedrock.invoke_model.call_args
        body = json.loads(call_args.kwargs["body"])
        assert body["messages"][0]["content"] == "What is an AI agent?"
