import boto3
from moto import mock_aws
import pytest
import json
import os



# Add this at the top of your existing test file
os.environ["DYNAMODB_TABLE"] = "test-conversations"
os.environ["AWS_DEFAULT_REGION"] = "us-east-1"

@mock_aws
def test_conversation_history_is_saved():
    """After invoking the handler, the message and response should be stored in DynamoDB."""
    from unittest.mock import patch, MagicMock
    
    # Create a real (mocked) DynamoDB table
    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    dynamodb.create_table(
        TableName="test-conversations",
        KeySchema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "session_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST"
    )

    mock_bedrock_response = {
        "body": MagicMock(read=lambda: json.dumps({
            "content": [{"text": "Hello! How can I help?"}]
        }).encode())
    }

    with patch("boto3.client") as mock_client:
        mock_client.return_value.invoke_model.return_value = mock_bedrock_response
        from  hello_world.app import lambda_handler 
        event = {"session_id": "abc123", "message": "Hello"}
        lambda_handler(event, {})

    table = dynamodb.Table("test-conversations")
    result = table.get_item(Key={"session_id": "abc123"})
    assert "Item" in result
    assert result["Item"]["session_id"] == "abc123"


@mock_aws
def test_previous_history_is_included_in_prompt():
    """If history exists for a session, it should be passed to Bedrock."""
    from unittest.mock import patch, MagicMock

    dynamodb = boto3.resource("dynamodb", region_name="us-east-1")
    table = dynamodb.create_table(
        TableName="test-conversations",
        KeySchema=[{"AttributeName": "session_id", "KeyType": "HASH"}],
        AttributeDefinitions=[{"AttributeName": "session_id", "AttributeType": "S"}],
        BillingMode="PAY_PER_REQUEST"
    )
    # Pre-seed existing history
    table.put_item(Item={
        "session_id": "abc123",
        "history": [
            {"role": "user", "content": "What is 2+2?"},
            {"role": "assistant", "content": "4"}
        ]
    })

    mock_bedrock_response = {
        "body": MagicMock(read=lambda: json.dumps({
            "content": [{"text": "Sure, continuing our chat!"}]
        }).encode())
    }

    with patch("boto3.client") as mock_client:
        mock_invoke = mock_client.return_value.invoke_model
        mock_invoke.return_value = mock_bedrock_response
        from  hello_world.app import lambda_handler 
        event = {"session_id": "abc123", "message": "And what is 3+3?"}
        lambda_handler(event, {})

        # Check that the Bedrock call included the previous history
        call_args = mock_invoke.call_args
        body = json.loads(call_args[1]["body"])
        messages = body["messages"]
        assert messages[0]["content"] == "What is 2+2?"