"""Test auto_instrument for Instructor."""

import instructor
import openai
from braintrust.integrations.test_utils import run_auto_smoke
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


def _call(memory_logger):
    # Drive a real instructor.from_openai call against a recorded cassette and
    # verify a parent task-typed Instructor span shows up alongside the OpenAI
    # llm child span. Cassette is shared with the in-process test suite under
    # integrations/instructor/cassettes/<version>/.
    client = openai.OpenAI(api_key="sk-test-dummy-api-key-for-vcr-tests")
    patched = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
    result = patched.chat.completions.create(
        model="gpt-4o-mini",
        response_model=Person,
        max_retries=3,
        messages=[{"role": "user", "content": "Extract Grace, age 45."}],
    )
    assert isinstance(result, Person)
    assert result.model_dump() == {"name": "Grace", "age": 45}

    raw = memory_logger.pop()
    spans = []
    for s in raw:
        if isinstance(s, list):
            spans.extend(s)
        else:
            spans.append(s)
    types = [s["span_attributes"].get("type") for s in spans]
    assert "task" in types, f"missing instructor parent (task) span: {types}"
    assert "llm" in types, f"missing openai child (llm) span: {types}"


run_auto_smoke(
    "instructor",
    cassette="TestInstructorOpenAISpans.test_instructor_openai_single_success",
    integration="instructor",
    run=_call,
)
print("SUCCESS")
