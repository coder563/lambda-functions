import requests
def handler(event, context):
   
    response = requests.get("https://httpbin.org/get")
    print("Response from httpbin.org:", response.text)
    return {
        'statusCode': 200,
        'body': 'Hello, This is my Lambda function!',
        'response_body': response.text
    }