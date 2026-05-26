"""Braintrust integration for the Instructor structured-output library."""

from typing import Any

from .integration import InstructorIntegration
from .patchers import InstructorPatcher


def wrap_instructor(client: Any) -> Any:
    """Instrument an ``instructor.Instructor`` / ``AsyncInstructor`` client in place.

    The Instructor client returned by ``instructor.from_openai(...)`` (and
    the other ``from_<provider>`` factories) is mutated to emit a Braintrust
    ``task``-typed span per ``create``/``create_with_completion``/
    ``create_partial``/``create_iterable`` call.  The underlying provider
    client is left untouched — combine with ``wrap_openai`` /
    ``wrap_anthropic`` / ``auto_instrument`` to also see the LLM child span.

    Returns *client* for chaining.
    """
    InstructorPatcher.wrap_target(type(client))
    return client


__all__ = [
    "InstructorIntegration",
    "wrap_instructor",
]
