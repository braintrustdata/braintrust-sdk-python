"""Instructor integration orchestration."""

from braintrust.integrations.base import BaseIntegration

from .patchers import InstructorPatcher


class InstructorIntegration(BaseIntegration):
    """Braintrust instrumentation for the Instructor structured-output library.

    Wraps ``Instructor.create*`` / ``AsyncInstructor.create*`` to emit one
    parent ``task``-typed span per call carrying Instructor-only metadata
    (``response_model``, ``mode``, ``max_retries``, ``retry_count``,
    ``validation_errors``).  Token usage is **not** logged here — the
    existing provider integrations (OpenAI, Anthropic, Cohere, …) own the
    ``llm``-typed child span for the underlying HTTP call.
    """

    name = "instructor"
    import_names = ("instructor",)
    # 1.11.0 has the stable ``instructor.core.client`` surface patched here.
    min_version = "1.11.0"
    patchers = (InstructorPatcher,)
