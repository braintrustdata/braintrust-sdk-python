"""CrewAI patchers — one ``CallbackPatcher`` that registers the listener."""

from typing import Any, ClassVar

from braintrust.integrations.base import CallbackPatcher


# Module-level cache for the single registered listener instance.  Keeping
# it at module scope (rather than on the integration class) means
# ``setup_crewai`` / ``patch_crewai`` / ``auto_instrument`` all see the
# same ``BraintrustCrewAIListener`` regardless of entry point.
_LISTENER: Any | None = None


def _unregister_event_handler(event_bus: Any, event_type: Any, handler: Any) -> None:
    """Best-effort wrapper around ``CrewAIEventsBus.off``.

    Pylint cannot currently infer the dynamically-generated event-bus API in
    CrewAI, so we use ``getattr`` here instead of calling ``off`` directly.
    """
    unregister = getattr(event_bus, "off", None)
    if callable(unregister):
        unregister(event_type, handler)


def _register_braintrust_listener() -> bool:
    """Idempotently create and register the Braintrust listener.

    The listener subclasses :class:`crewai.events.BaseEventListener`, whose
    ``__init__`` registers handlers on the process-singleton
    ``crewai_event_bus``.  We cache the instance at module scope so repeat
    calls (e.g. ``setup_crewai()`` followed by ``auto_instrument()``) do
    not register a second listener.
    """
    global _LISTENER  # noqa: PLW0603
    if _LISTENER is not None:
        return True

    # Lazy import: CrewAI may not be installed in the environment.
    from .tracing import BraintrustCrewAIListener

    _LISTENER = BraintrustCrewAIListener()
    return True


def _listener_registered() -> bool:
    """Return whether the Braintrust listener is currently registered."""
    return _LISTENER is not None


def _get_registered_listener() -> Any | None:
    """Return the registered listener, or ``None`` when setup has not run."""
    return _LISTENER


def _reset_for_testing() -> None:
    """Unregister the Braintrust listener and forget cached runtime state.

    Intended for pytest fixtures that need to restart from a clean slate.
    Safe to call when CrewAI is not importable or nothing has been
    registered. Not part of the public API.
    """
    global _LISTENER  # noqa: PLW0603
    if _LISTENER is None:
        return

    try:
        from crewai.events.event_bus import crewai_event_bus
    except ImportError:
        _LISTENER = None
        return

    for event_type, handlers in list(crewai_event_bus._sync_handlers.items()):
        for handler in list(handlers):
            handler_mod = getattr(handler, "__module__", "")
            if "braintrust" in handler_mod and "crewai" in handler_mod:
                _unregister_event_handler(crewai_event_bus, event_type, handler)

    _LISTENER = None

    # Clear the runtime subclass cache so the next setup rebuilds it; this
    # matters for tests that monkey-patch the listener base class.
    from .tracing import BraintrustCrewAIListener

    BraintrustCrewAIListener._cls = None


class EventBusPatcher(CallbackPatcher):
    """Register :class:`BraintrustCrewAIListener` on ``crewai_event_bus``.

    The target module check gates this patcher on the event-bus module being
    importable, not the top-level ``crewai`` package.  That means users who
    install only a CrewAI fork missing the event-bus surface get a clean
    skip rather than an import error during setup.
    """

    name: ClassVar[str] = "crewai.event_bus"
    target_module: ClassVar[str] = "crewai.events.event_bus"
    callback: ClassVar[Any] = staticmethod(_register_braintrust_listener)
    state_getter: ClassVar[Any] = staticmethod(_listener_registered)
