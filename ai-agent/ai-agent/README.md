# ai-agent

A serverless AI agent built with AWS Lambda, Amazon Bedrock (Claude), and AWS SAM.

## What it does

Exposes a Lambda function that accepts a prompt and returns a response from Claude via Amazon Bedrock. Supports multi-turn conversation history stored in DynamoDB.

## Project structure

```
hello_world/        Lambda function code
tests/unit/         Unit tests (pytest + moto)
template.yaml       SAM infrastructure definition
.github/workflows/  CI/CD pipeline (test + deploy on push to main)
```

## Event format

**Simple prompt:**
```json
{ "prompt": "What is an AI agent?" }
```

**With conversation session:**
```json
{ "session_id": "abc123", "message": "What is an AI agent?" }
```

When `session_id` is provided, the handler loads previous conversation history from DynamoDB, sends the full history to Claude, and saves the updated history back.

## Environment variables

| Variable | Description |
|---|---|
| `DYNAMODB_TABLE` | DynamoDB table name for conversation history (optional) |

## Local development

**Prerequisites:** AWS CLI, SAM CLI, Python 3.12, Docker

```bash
# Install test dependencies
pip install pytest boto3 moto

# Run tests
python3 -m pytest tests/unit/ -v

# Build and invoke locally
sam build
sam local invoke HelloWorldFunction --event events/event.json
```

## Deployment

Push to `main` — GitHub Actions runs tests then deploys automatically via `sam deploy`.

**Required GitHub secrets:**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

## Infrastructure

- **Runtime:** Python 3.12
- **Region:** us-east-1
- **Model:** Claude 3.5 Sonnet v2 (`us.anthropic.claude-3-5-sonnet-20241022-v2:0`)
- **IAM:** Lambda role with `bedrock:InvokeModel` permission
