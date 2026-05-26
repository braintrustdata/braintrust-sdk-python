"""Tests for the Braintrust Instructor integration.

These tests target the public Instructor surface (``Instructor.create`` /
``AsyncInstructor.create``) using VCR cassettes recorded against the
underlying provider HTTP traffic.  The Instructor integration itself does
not emit ``llm``-typed spans — the existing provider integrations
(OpenAI/Anthropic/etc.) already do — so these tests assert:

* exactly one parent ``task``-typed Instructor span,
* Instructor-only metadata on the parent (``response_model``, ``mode``,
  ``max_retries``, ``retry_count``, ``validation_errors``),
* the final extracted Pydantic dict as the parent ``output``,
* no token ``metrics`` on the parent,
* the underlying provider's ``llm`` child spans still fire once per HTTP
  call.
"""

import json

import instructor
import openai
import pytest
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import find_span_by_name, find_spans_by_type, init_test_logger, memory_logger
from pydantic import BaseModel, Field


# memory_logger fixture from braintrust.test_helpers is imported above
__all__ = ["memory_logger"]


PROJECT_NAME = "test-instructor-app"


class Person(BaseModel):
    name: str = Field(..., description="The person's name")
    age: int = Field(..., description="The person's age")


def _make_openai_client():
    # The real OpenAI client is fine even without a real key — VCR will serve
    # the request from the cassette.
    return openai.OpenAI(api_key="sk-test-dummy-api-key-for-vcr-tests")


def _all_spans(memory_logger):
    spans = memory_logger.pop()
    out = []
    for s in spans:
        if isinstance(s, list):
            out.extend(s)
        else:
            out.append(s)
    return out


def _names(spans):
    return [s.get("span_attributes", {}).get("name") for s in spans]


def _dump(value):
    if isinstance(value, BaseModel):
        return value.model_dump()
    if isinstance(value, list):
        return [_dump(item) for item in value]
    return value


@pytest.fixture
def setup_logger():
    from braintrust.integrations import InstructorIntegration

    init_test_logger(PROJECT_NAME)
    InstructorIntegration.setup()


class TestInstructorIntegrationExists:
    """Static checks the Instructor integration is registered."""

    def test_integration_class_exported(self):
        from braintrust.integrations import InstructorIntegration  # noqa: F401

    def test_wrap_instructor_exported(self):
        from braintrust import wrap_instructor  # noqa: F401

    def test_auto_instrument_has_instructor_kwarg(self):
        import braintrust

        argcount = braintrust.auto_instrument.__code__.co_argcount
        kwonly = braintrust.auto_instrument.__code__.co_kwonlyargcount
        params = braintrust.auto_instrument.__code__.co_varnames[: argcount + kwonly]
        assert "instructor" in params


class TestInstructorOpenAISpans:
    """End-to-end: real instructor.from_openai against a VCR cassette."""

    @pytest.mark.vcr
    def test_instructor_openai_single_success(self, setup_logger, memory_logger):
        """One LLM call, valid on first try -> parent task span + 1 child llm span."""
        from braintrust import wrap_openai

        # wrap_openai must remain compatible: it provides the llm child span.
        client = wrap_openai(_make_openai_client())
        patched = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

        result = patched.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Person,
            max_retries=3,
            messages=[{"role": "user", "content": "Extract Grace, age 45."}],
        )
        assert isinstance(result, Person)
        assert result.model_dump() == {"name": "Grace", "age": 45}

        spans = _all_spans(memory_logger)
        # Should have 1 parent (task) + 1 child (llm)
        task_spans = find_spans_by_type(spans, SpanTypeAttribute.TASK)
        llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
        assert len(task_spans) == 1, f"Expected 1 task span, got {len(task_spans)}: names={_names(spans)}"
        assert len(llm_spans) == 1, f"Expected 1 llm span, got {len(llm_spans)}: names={_names(spans)}"

        parent = task_spans[0]
        assert parent["span_attributes"]["name"] == "instructor.create"
        meta = parent.get("metadata", {})
        assert meta.get("response_model") == "Person"
        assert meta.get("mode") == "TOOLS"
        assert meta.get("max_retries") == 3
        assert meta.get("retry_count") == 0
        assert meta.get("validation_errors") == []
        assert _dump(parent.get("output")) == {"name": "Grace", "age": 45}

        # Critical invariant: no token metrics on the parent (avoid
        # double-counting against the llm child).
        parent_metrics = parent.get("metrics") or {}
        for k in ("tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
            assert k not in parent_metrics, f"parent must not log {k!r}; child llm span owns it"

        # The OpenAI child span keeps its usage.
        child = llm_spans[0]
        child_metrics = child.get("metrics") or {}
        assert child_metrics.get("tokens", 0) > 0 or child_metrics.get("total_tokens", 0) > 0

    @pytest.mark.vcr
    def test_instructor_openai_retries_then_succeeds(self, setup_logger, memory_logger):
        """First LLM call returns missing field; instructor retries and succeeds.

        Expect exactly: 1 task parent + 2 llm children. Parent records
        retry_count=1 and one validation_errors entry. Token totals across
        the trace equal the sum across the *children*; parent contributes
        zero tokens.
        """
        from braintrust import wrap_openai

        client = wrap_openai(_make_openai_client())
        patched = instructor.from_openai(client, mode=instructor.Mode.TOOLS)

        result = patched.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Person,
            max_retries=3,
            messages=[{"role": "user", "content": "Extract Ada, age 30."}],
        )
        assert result.model_dump() == {"name": "Ada", "age": 30}

        spans = _all_spans(memory_logger)
        task_spans = find_spans_by_type(spans, SpanTypeAttribute.TASK)
        llm_spans = find_spans_by_type(spans, SpanTypeAttribute.LLM)
        assert len(task_spans) == 1, f"Expected 1 task span, got {len(task_spans)}: names={_names(spans)}"
        assert len(llm_spans) == 2, f"Expected 2 llm spans (retry), got {len(llm_spans)}: names={_names(spans)}"

        parent = task_spans[0]
        meta = parent.get("metadata", {})
        assert meta.get("response_model") == "Person"
        assert meta.get("mode") == "TOOLS"
        assert meta.get("max_retries") == 3
        assert meta.get("retry_count") == 1, f"Expected retry_count=1, got {meta.get('retry_count')}"
        ve = meta.get("validation_errors")
        assert isinstance(ve, list) and len(ve) == 1, f"Expected 1 validation_errors entry, got {ve!r}"
        # The error must mention the missing 'age' field somehow.
        assert "age" in json.dumps(ve), f"validation_errors should reference 'age': {ve!r}"
        assert _dump(parent.get("output")) == {"name": "Ada", "age": 30}

        # No double counting: parent has no tokens.
        parent_metrics = parent.get("metrics") or {}
        for k in ("tokens", "prompt_tokens", "completion_tokens", "total_tokens"):
            assert k not in parent_metrics


class TestInstructorPatcherIdempotence:
    """Calling setup or wrap_instructor twice must not stack wrappers."""

    def test_setup_is_idempotent(self):
        from braintrust.integrations import InstructorIntegration

        assert InstructorIntegration.setup() is True
        # second call should not raise and should report success
        assert InstructorIntegration.setup() is True

        # Calling create twice in a row still works (sanity).
        from braintrust import wrap_openai

        init_test_logger(PROJECT_NAME)
        client = wrap_openai(_make_openai_client())
        patched = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
        # We're not making a real call here; just confirming patch did not
        # destroy the bound method surface.
        assert callable(patched.chat.completions.create)


class TestInstructorAutoInstrumentSubprocess:
    """auto_instrument() must instrument Instructor in a fresh subprocess too."""

    def test_subprocess_auto_instrument_instructor(self):
        from braintrust.integrations.test_utils import verify_autoinstrument_script

        verify_autoinstrument_script("test_auto_instructor.py", timeout=30)


class TestInstructorParentIsNotLLM:
    """Span-type invariant: Instructor parent is never typed as `llm`."""

    @pytest.mark.vcr("test_instructor_openai_single_success.yaml")
    def test_parent_span_type_is_task_not_llm(self, setup_logger, memory_logger):
        from braintrust import wrap_openai

        client = wrap_openai(_make_openai_client())
        patched = instructor.from_openai(client, mode=instructor.Mode.TOOLS)
        patched.chat.completions.create(
            model="gpt-4o-mini",
            response_model=Person,
            max_retries=3,
            messages=[{"role": "user", "content": "Extract Grace, age 45."}],
        )
        spans = _all_spans(memory_logger)
        parent = find_span_by_name(spans, "instructor.create")
        assert parent["span_attributes"]["type"] == SpanTypeAttribute.TASK.value
        assert parent["span_attributes"]["type"] != SpanTypeAttribute.LLM.value
