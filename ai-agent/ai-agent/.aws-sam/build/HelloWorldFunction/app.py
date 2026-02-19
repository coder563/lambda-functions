import json
import boto3

# import requests


def lambda_handler(event, context):
    """Sample pure Lambda function

    Parameters
    ----------
    event: dict, required
        API Gateway Lambda Proxy Input Format

        Event doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html#api-gateway-simple-proxy-for-lambda-input-format

    context: object, required
        Lambda Context runtime methods and attributes

        Context doc: https://docs.aws.amazon.com/lambda/latest/dg/python-context-object.html

    Returns
    ------
    API Gateway Lambda Proxy Output Format: dict

        Return doc: https://docs.aws.amazon.com/apigateway/latest/developerguide/set-up-lambda-proxy-integrations.html
    """

    # try:
    #     ip = requests.get("http://checkip.amazonaws.com/")
    # except requests.RequestException as e:
    #     # Send some context about this error to Lambda Logs
    #     print(e)

    #     raise e
    prompt = event.get("prompt", "Say hello in one sentence.")
    # create a Bedrock client
    client = boto3.client("bedrock-runtime", region_name="us-east-1")

    #Call Claude 2 with the prompt
    response = client.invoke_model(
        modelId="us.anthropic.claude-3-5-sonnet-20241022-v2:0",
        body=json.dumps({
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 512,
            "messages": [
                { "role": "user", "content": prompt }
            ]                
        })
    )
    response_body = json.loads(response["body"].read())
    answer = response_body["content"][0]["text"]
    return {
        "statusCode": 200,
        "body": answer
        }
    
