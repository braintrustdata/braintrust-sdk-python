"""Test auto_instrument for Instructor."""

import instructor
import openai
from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context
from pydantic import BaseModel


class Person(BaseModel):
    name: str
    age: int


# 1. Instrument
results = auto_instrument()
assert results.get("instructor") is True, results

# 2. Idempotent
results2 = auto_instrument()
assert results2.get("instructor") is True

# 3. Drive a real instructor.from_openai call against a recorded cassette and
#    verify a parent task-typed Instructor span shows up alongside the OpenAI
#    llm child span.  Cassette is shared with the in-process test suite under
#    integrations/instructor/cassettes/<version>/.
with autoinstrument_test_context(
    "TestInstructorOpenAISpans.test_instructor_openai_single_success",
    integration="instructor",
) as memory_logger:
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
    parent = next(s for s in spans if s["span_attributes"].get("type") == "task")
    assert parent["span_attributes"]["name"] == "instructor.create"
    assert parent["metadata"]["response_model"] == "Person"
    assert parent["metadata"]["mode"] == "TOOLS"
    assert parent["metadata"]["retry_count"] == 0

print("SUCCESS")
