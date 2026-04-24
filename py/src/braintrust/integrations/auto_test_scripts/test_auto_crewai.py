"""Test auto_instrument for CrewAI.

Verifies that ``auto_instrument(crewai=True)`` registers the Braintrust
CrewAI listener on ``crewai_event_bus`` and is idempotent across repeated
calls.  Full span-shape coverage lives in ``test_crewai.py``.
"""

# pylint: disable=import-error

from braintrust.auto import auto_instrument
from braintrust.integrations.crewai import BraintrustCrewAIListener
from braintrust.integrations.crewai.patchers import _get_registered_listener


# 1. Not registered initially.
assert _get_registered_listener() is None

# 2. Instrument once.
results = auto_instrument()
assert results.get("crewai") is True
listener1 = _get_registered_listener()
assert listener1 is not None
assert isinstance(listener1, BraintrustCrewAIListener)

# 3. Idempotent — same listener, still reports True.
results2 = auto_instrument()
assert results2.get("crewai") is True
assert _get_registered_listener() is listener1

# 4. Listener is actually subscribed on the CrewAI event bus.
from crewai.events import CrewKickoffStartedEvent
from crewai.events.event_bus import crewai_event_bus


sync_handlers = crewai_event_bus._sync_handlers.get(CrewKickoffStartedEvent, frozenset())
assert sync_handlers, "Expected at least one sync handler registered for CrewKickoffStartedEvent"

print("SUCCESS")
