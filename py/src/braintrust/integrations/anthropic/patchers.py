from braintrust.integrations.base import FunctionWrapperPatcher

from .tracing import _anthropic_init_wrapper, _async_anthropic_init_wrapper


class AnthropicSyncInitPatcher(FunctionWrapperPatcher):
    name = "anthropic.init.sync"
    target_path = "Anthropic.__init__"
    wrapper = _anthropic_init_wrapper


class AnthropicAsyncInitPatcher(FunctionWrapperPatcher):
    name = "anthropic.init.async"
    target_path = "AsyncAnthropic.__init__"
    wrapper = _async_anthropic_init_wrapper
