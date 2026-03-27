"""DSPy patchers — one patcher per coherent patch target."""

from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import _configure_wrapper


class DSPyConfigurePatcher(FunctionWrapperPatcher):
    """Patch ``dspy.configure`` to auto-add ``BraintrustDSpyCallback``."""

    name = "dspy.configure"
    target_path = "configure"
    wrapper = _configure_wrapper


# ---------------------------------------------------------------------------
# Public helper
# ---------------------------------------------------------------------------


def patch_dspy() -> bool:
    """
    Patch DSPy to automatically add Braintrust tracing callback.

    After calling this, all calls to dspy.configure() will automatically
    include the BraintrustDSpyCallback.

    Returns:
        True if DSPy was patched (or already patched), False if DSPy is not installed.

    Example:
        ```python
        import braintrust
        braintrust.patch_dspy()

        import dspy
        lm = dspy.LM("openai/gpt-4o-mini")
        dspy.configure(lm=lm)  # BraintrustDSpyCallback auto-added!
        ```
    """
    from .integration import DSPyIntegration

    return DSPyIntegration.setup()
