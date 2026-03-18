# ai-agent

A serverless AI agent built with AWS Lambda, Amazon Bedrock (Claude), and AWS SAM. Includes both a hand-rolled agent and a LangChain-based agent implementing the same behaviour.

---

## How it works

### The agent loop (ReAct pattern)

The agent follows the **ReAct** pattern — Reasoning + Acting:

```
User sends message
       │
       ▼
  Send to Claude
       │
       ▼
 stop_reason == "tool_use"?
    ├── YES → execute the tool → feed result back → send to Claude again
    └── NO  → return final answer to user
```

This loop repeats until Claude returns a final answer (`stop_reason: end_turn`) or the iteration limit is hit (max 10).

---

## Two implementations

### 1. Hand-rolled (`hello_world/app.py`)

Built from scratch using `boto3` directly against Amazon Bedrock.

| Concept | Implementation |
|---|---|
| LLM call | `boto3.client("bedrock-runtime").invoke_model()` |
| Tool definitions | `TOOLS` — a list of JSON schemas describing each tool |
| Tool execution | `execute_tool()` — dispatches to the right Python function |
| Agent loop | `run_agent()` — `for _ in range(MAX_ITERATIONS)` loop |
| Conversation memory | `load_history()` / `save_history()` via DynamoDB |
| Entry point | `lambda_handler()` — orchestrates all of the above |

**Flow:**

```
lambda_handler()
    └── load_history()          # fetch past messages from DynamoDB
    └── run_agent()             # ReAct loop
            └── invoke_model()  # call Claude via Bedrock
            └── execute_tool()  # run tool if Claude requests it
    └── save_history()          # persist updated messages to DynamoDB
    └── return answer
```

### 2. LangChain (`hello_world/agent_langchain.py`)

Same agent rebuilt using the LangChain framework (`langchain-aws`, `langgraph`).

| Concept | Implementation |
|---|---|
| LLM call | `ChatBedrock` — wraps boto3 internally |
| Tool definitions | `@tool` decorator — reads function signature + docstring |
| Tool execution | Handled by the framework |
| Agent loop | `create_react_agent()` from `langgraph.prebuilt` |
| Entry point | `run_langchain_agent()` |

**Flow:**

```
run_langchain_agent()
    └── ChatBedrock(...)             # LLM client
    └── create_react_agent(llm, tools)  # builds the ReAct loop
    └── agent.invoke({"messages": [...]})  # runs the full loop internally
    └── extract last AIMessage → return answer
```

---

## Tools available

| Tool | Description |
|---|---|
| `get_weather(location)` | Returns current weather for a city |
| `calculator(expression)` | Evaluates a math expression (e.g. `"10 * 5"`) |
| `get_current_time()` | Returns the current date and time |

Claude decides which tool to call (if any) based on the user's message. The tool descriptions and schemas tell Claude what each tool does and what arguments it expects.

---

## Project structure

```
hello_world/
    app.py                  Hand-rolled agent (Lambda handler)
    agent_langchain.py      LangChain agent
    requirements.txt        Python dependencies

tests/unit/
    test_handler.py         Basic handler tests
    test_conversation.py    DynamoDB memory tests
    test_tool_calling.py    Tool-use flow tests
    test_agent_loop.py      Multi-step loop + iteration limit tests
    test_tools.py           Individual tool tests
    test_langchain_agent.py LangChain agent tests

template.yaml               SAM infrastructure (Lambda + DynamoDB + IAM)
.github/workflows/
    deploy.yml              CI/CD: test → build → deploy on push to main
```

---

## Event format

**Simple prompt:**
```json
{ "prompt": "What is the weather in London?" }
```

**With conversation session:**
```json
{ "session_id": "abc123", "message": "What is the weather in London?" }
```

When `session_id` is provided, the handler loads previous conversation history from DynamoDB, sends the full history to Claude, and saves the updated history back after each turn.

---

## Environment variables

| Variable | Description |
|---|---|
| `DYNAMODB_TABLE` | DynamoDB table name for conversation history (optional) |

---

## Local development

**Prerequisites:** AWS CLI, SAM CLI, Python 3.12, Docker

```bash
# Install dependencies
pip install pytest boto3 moto langchain langchain-aws langgraph

# Run all tests
python3 -m pytest tests/unit/ -v

# Build and invoke locally
sam build
sam local invoke HelloWorldFunction --event events/event.json
```

---

## Deployment

Push to `main` — GitHub Actions runs tests then deploys automatically via `sam deploy`.

**Required GitHub secrets:**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`

---

## Infrastructure

- **Runtime:** Python 3.12
- **Region:** us-east-1
- **Model:** Claude 3.5 Sonnet v2 (`us.anthropic.claude-3-5-sonnet-20241022-v2:0`)
- **IAM:** Lambda role with `bedrock:InvokeModel` and DynamoDB permissions