#!/usr/bin/env python
"""boto3 Bedrock Runtime client traced via braintrust.auto_instrument()."""

import os

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-bedrock")

import boto3  # noqa: E402


MODEL = os.getenv("BRAINTRUST_BEDROCK_MODEL", "us.amazon.nova-lite-v1:0")
REGION = os.getenv("AWS_REGION") or os.getenv("AWS_DEFAULT_REGION") or "us-east-1"


@braintrust.traced
def answer_question(question: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=REGION)
    response = client.converse(
        modelId=MODEL,
        messages=[{"role": "user", "content": [{"text": question}]}],
        inferenceConfig={"maxTokens": 64, "temperature": 0},
    )
    message = response["output"]["message"]
    return "".join(block.get("text", "") for block in message.get("content", []))


def main() -> None:
    answer = answer_question("What is the capital of Australia? Reply in one sentence.")
    print(answer)
    braintrust.flush()


if __name__ == "__main__":
    main()
