#!/usr/bin/env python

import braintrust


braintrust.auto_instrument()
braintrust.init_logger(project="example-openai-project")

from openai import OpenAI


client = OpenAI()


# @traced automatically logs the input (args) and output (return value)
# of this function to a span. To ensure the span is named `answer_question`,
# you should name the function `answer_question`.
@braintrust.traced
def answer_question(body: str) -> str:
    prompt = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": body},
    ]

    result = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=prompt,
        temperature=0.5,
    )
    return result.choices[0].message.content


def main():
    input_text = "What's the capital of Australia?"
    result = answer_question(input_text)
    print(result)


if __name__ == "__main__":
    main()
