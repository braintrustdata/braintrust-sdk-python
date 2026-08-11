"""Registration patcher for Vercel AI SDK telemetry."""

from typing import Any, ClassVar

from braintrust.integrations.base import CallbackPatcher

from .tracing import BraintrustAISDKAdapter


_ADAPTER: BraintrustAISDKAdapter | None = None


def _adapter_registered() -> bool:
    return _ADAPTER is not None


def _register_adapter() -> bool:
    global _ADAPTER  # noqa: PLW0603
    if _ADAPTER is not None:
        return True

    from ai import experimental_telemetry  # pylint: disable=import-error

    adapter = BraintrustAISDKAdapter()
    experimental_telemetry.register(adapter)
    _ADAPTER = adapter
    return True


def unregister_adapter() -> bool:
    """Unregister the process-wide Braintrust telemetry adapter."""
    global _ADAPTER  # noqa: PLW0603
    if _ADAPTER is None:
        return True

    from ai import experimental_telemetry  # pylint: disable=import-error

    adapter = _ADAPTER
    try:
        experimental_telemetry.unregister(adapter)
    except ValueError:
        pass
    _ADAPTER = None
    return True


class TelemetryAdapterPatcher(CallbackPatcher):
    """Register one Braintrust adapter with ``ai.experimental_telemetry``."""

    name: ClassVar[str] = "ai.experimental_telemetry.adapter"
    target_module: ClassVar[str] = "ai.experimental_telemetry"
    callback: ClassVar[Any] = staticmethod(_register_adapter)
    state_getter: ClassVar[Any] = staticmethod(_adapter_registered)
