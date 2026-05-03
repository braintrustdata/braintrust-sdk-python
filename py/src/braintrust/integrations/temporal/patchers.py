"""Temporal patchers and public helpers."""

from typing import Any

from braintrust.integrations.base import FunctionWrapperPatcher


def _plugin_classes() -> tuple[type[Any], type[Any]]:
    from .plugin import BraintrustInterceptor, BraintrustPlugin

    return BraintrustInterceptor, BraintrustPlugin


def _has_braintrust_plugin(plugin: Any) -> bool:
    BraintrustInterceptor, BraintrustPlugin = _plugin_classes()
    if isinstance(plugin, BraintrustPlugin):
        return True
    interceptors = []
    for attr in ("interceptors", "client_interceptors", "worker_interceptors"):
        interceptors.extend(getattr(plugin, attr, None) or [])
    return any(isinstance(interceptor, BraintrustInterceptor) for interceptor in interceptors)


def _with_braintrust_plugin(plugins: Any | None) -> list[Any]:
    """Return plugins with a Braintrust plugin appended unless already present."""
    plugin_list = list(plugins or [])
    if any(_has_braintrust_plugin(plugin) for plugin in plugin_list):
        return plugin_list
    _, BraintrustPlugin = _plugin_classes()
    return [*plugin_list, BraintrustPlugin()]


def _client_connect_wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    kwargs["plugins"] = _with_braintrust_plugin(kwargs.get("plugins"))
    return wrapped(*args, **kwargs)


def _worker_init_wrapper(wrapped: Any, instance: Any, args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    kwargs["plugins"] = _with_braintrust_plugin(kwargs.get("plugins"))
    return wrapped(*args, **kwargs)


class ClientConnectPatcher(FunctionWrapperPatcher):
    """Patch Temporal client connections to install BraintrustPlugin."""

    name = "temporal.client.connect"
    target_module = "temporalio.client"
    target_path = "Client.connect"
    wrapper = _client_connect_wrapper


class WorkerInitPatcher(FunctionWrapperPatcher):
    """Patch Temporal workers to install BraintrustPlugin."""

    name = "temporal.worker.init"
    target_module = "temporalio.worker"
    target_path = "Worker.__init__"
    wrapper = _worker_init_wrapper
