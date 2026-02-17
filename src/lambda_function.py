import requests
def handler(event, context):
   
    response = requests.get("https://httpbin.org/get")
    return {
        'statusCode': 200,
        'body': 'Hello, This is my Lambda function!'
    }