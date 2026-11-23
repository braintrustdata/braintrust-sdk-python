"""Registration patcher for Vercel AI SDK telemetry."""

from typing import Any, ClassVar

from braintrust.integrations.base import CallbackPatcher, _import_first_available

from .tracing import BraintrustAISDKAdapter


_ADAPTER: BraintrustAISDKAdapter | None = None


def _import_telemetry() -> Any | None:
    return _import_first_available(("ai.telemetry", "ai.experimental_telemetry"))


def _adapter_registered() -> bool:
    return _ADAPTER is not None


def _register_adapter() -> bool:
    global _ADAPTER  # noqa: PLW0603
    if _ADAPTER is not None:
        return True

    telemetry = _import_telemetry()
    if telemetry is None:
        return False

    adapter = BraintrustAISDKAdapter()
    telemetry.register(adapter)
    _ADAPTER = adapter
    return True


def unregister_adapter() -> bool:
    """Unregister the process-wide Braintrust telemetry adapter."""
    global _ADAPTER  # noqa: PLW0603
    if _ADAPTER is None:
        return True

    telemetry = _import_telemetry()
    if telemetry is None:
        return False

    adapter = _ADAPTER
    try:
        telemetry.unregister(adapter)
    except ValueError:
        pass
    _ADAPTER = None
    return True


class TelemetryAdapterPatcher(CallbackPatcher):
    """Register one Braintrust adapter with the AI SDK telemetry module."""

    # Keep the existing patcher identifier across AI SDK versions.
    name: ClassVar[str] = "ai.experimental_telemetry.adapter"
    callback: ClassVar[Any] = staticmethod(_register_adapter)
    state_getter: ClassVar[Any] = staticmethod(_adapter_registered)
