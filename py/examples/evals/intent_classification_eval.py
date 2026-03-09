from braintrust import Eval
from openai import OpenAI

client = OpenAI()

DATASET = [
    {
        "input": "What's your return policy?",
        "expected": "policy_question",
    },
    {
        "input": "I need help with my order",
        "expected": "support_request",
    },
]


def task(input):
    response = client.responses.create(
        model="gpt-5-mini",
        input=[{"role": "user", "content": input}],
    )
    return response.output_text


def intent_classifier(input, output, expected, metadata):
    keywords = {
        "policy_question": ["policy", "return", "refund", "warranty"],
        "support_request": ["help", "issue", "problem", "support"],
        "product_inquiry": ["price", "feature", "available", "buy"],
    }

    for intent, words in keywords.items():
        if any(word in input.lower() for word in words):
            return intent

    return "other"


Eval(
    "Intent Classification",
    data=DATASET,
    task=task,
    scores=[intent_classifier],
)
