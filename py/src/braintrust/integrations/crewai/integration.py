"""CrewAI integration — orchestration class.

Registers a single event-bus listener on :data:`crewai.events.crewai_event_bus`
that maps CrewAI events (crew kickoff, tasks, agent execution, LLM calls,
tool usage) into Braintrust spans.

Requires CrewAI 1.13.0 or newer, which is the first release that exposes
the full causal-id surface (``event_id``, ``parent_event_id``,
``started_event_id``) plus the ``usage`` field on
``LLMCallCompletedEvent`` the integration consumes.
"""

from braintrust.integrations.base import BaseIntegration

from .patchers import EventBusPatcher


class CrewAIIntegration(BaseIntegration):
    """Braintrust instrumentation for CrewAI."""

    name = "crewai"
    import_names = ("crewai",)
    # 1.13.0 is the first release with the event-bus surface this
    # integration depends on (``started_event_id`` on BaseEvent + ``usage``
    # on ``LLMCallCompletedEvent``).  Older 1.x releases ship the bus but
    # not these fields, so we gate instead of trying to version-branch the
    # listener.
    min_version = "1.13.0"
    patchers = (EventBusPatcher,)
