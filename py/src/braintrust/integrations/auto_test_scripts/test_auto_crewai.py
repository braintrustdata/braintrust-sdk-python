"""Test auto_instrument for CrewAI.

Verifies that ``auto_instrument(crewai=True)`` registers the Braintrust
CrewAI listener on ``crewai_event_bus`` and is idempotent across repeated
calls.  Full span-shape coverage lives in ``test_crewai.py``.
"""

# pylint: disable=import-error

from braintrust.integrations.crewai import BraintrustCrewAIListener
from braintrust.integrations.crewai.patchers import _get_registered_listener
from braintrust.integrations.test_utils import run_auto_smoke


def _is_patched() -> bool:
    listener = _get_registered_listener()
    return isinstance(listener, BraintrustCrewAIListener)


run_auto_smoke("crewai", is_patched=_is_patched)

# Listener stays the same across the two auto_instrument calls.
listener = _get_registered_listener()
assert listener is not None
assert isinstance(listener, BraintrustCrewAIListener)

# Listener is actually subscribed on the CrewAI event bus.
from crewai.events import CrewKickoffStartedEvent
from crewai.events.event_bus import crewai_event_bus


sync_handlers = crewai_event_bus._sync_handlers.get(CrewKickoffStartedEvent, frozenset())
assert sync_handlers, "Expected at least one sync handler registered for CrewKickoffStartedEvent"

print("SUCCESS")
