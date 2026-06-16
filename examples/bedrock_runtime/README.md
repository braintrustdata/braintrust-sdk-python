# AWS Bedrock + Braintrust

Calls `braintrust.auto_instrument()` to patch boto3's Bedrock Runtime client factory, then runs a single `client.converse(...)` call against Amazon Nova Lite. The trace captures a parent `answer_question` span plus a child `bedrock.converse` LLM span with input, output, metadata, and token metrics.

## Run

```bash
export BRAINTRUST_API_KEY=...
export AWS_BEARER_TOKEN_BEDROCK=...
export AWS_REGION=us-east-1
export AWS_DEFAULT_REGION=us-east-1

uv sync
uv run python example.py
```

If you use IAM credentials instead of a Bedrock API key, set the usual AWS variables instead:

```bash
export AWS_ACCESS_KEY_ID=...
export AWS_SECRET_ACCESS_KEY=...
export AWS_SESSION_TOKEN=... # if using temporary credentials
```

The default model is `us.amazon.nova-lite-v1:0`. Override it with `BRAINTRUST_BEDROCK_MODEL` if needed.
