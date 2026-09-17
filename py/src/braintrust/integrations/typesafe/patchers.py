"""Patchers for TypeSafe sync and async System One calls."""

from braintrust.integrations.base import CompositeFunctionWrapperPatcher, FunctionWrapperPatcher

from .tracing import _async_system_one_wrapper, _system_one_wrapper


class SystemOnePatcher(FunctionWrapperPatcher):
    name = "typesafe.system_one"
    target_module = "typesafe_sdk._core.client.sync.client"
    target_path = "TypeSafeClient.system_one"
    wrapper = _system_one_wrapper


class AsyncSystemOnePatcher(FunctionWrapperPatcher):
    name = "typesafe.async.system_one"
    target_module = "typesafe_sdk._core.client.aio.client"
    target_path = "AsyncTypeSafeClient.system_one"
    wrapper = _async_system_one_wrapper


class TypeSafePatcher(CompositeFunctionWrapperPatcher):
    name = "typesafe.system_one.all"
    sub_patchers = (SystemOnePatcher, AsyncSystemOnePatcher)
