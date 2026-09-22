import importlib.machinery
import importlib.util
import sys
import types

from braintrust.integrations.base import _import_optional_module, _resolve_attr_path


def _make_lazy_module(name, exports):
    """Build a module that only exposes *exports* through PEP 562 ``__getattr__``.

    This mirrors how openai >= 3.16.1 ships its resource packages: the names are
    declared in ``__all__`` but are only imported on first attribute access.
    """
    module = types.ModuleType(name)
    module.__all__ = list(exports)
    module.__getattr__ = lambda attr: exports[attr] if attr in exports else _raise(attr, name)
    return module


def _raise(attr, name):
    raise AttributeError(f"module {name!r} has no attribute {attr!r}")


class _Resource:
    def create(self):
        return "created"


def test_resolve_attr_path_follows_lazy_module_exports():
    module = _make_lazy_module("fake_sdk.resources", {"Resource": _Resource})

    assert _resolve_attr_path(module, "Resource") is _Resource
    assert _resolve_attr_path(module, "Resource.create") is _Resource.__dict__["create"]


def test_resolve_attr_path_returns_none_for_missing_lazy_export():
    module = _make_lazy_module("fake_sdk.resources", {"Resource": _Resource})

    assert _resolve_attr_path(module, "Missing") is None
    assert _resolve_attr_path(module, "Resource.missing") is None


def test_resolve_attr_path_does_not_invoke_descriptors_on_classes():
    class WithProperty:
        @property
        def value(self):  # pragma: no cover - must never be invoked
            raise AssertionError("property should not be evaluated")

    resolved = _resolve_attr_path(WithProperty, "value")
    assert isinstance(resolved, property)


def test_resolve_attr_path_walks_real_submodules():
    assert _resolve_attr_path(sys.modules["braintrust.integrations.base"], "BasePatcher") is not None


def test_import_optional_module_prefers_sys_modules(monkeypatch):
    """An already-imported module must not go back through importlib.

    Re-acquiring the import lock for every patcher on every setup() is what
    trips CPython 3.10's re-entrancy bookkeeping.
    """
    sentinel = types.ModuleType("braintrust_fake_sdk")
    monkeypatch.setitem(sys.modules, "braintrust_fake_sdk", sentinel)

    def explode(name):  # pragma: no cover - must never be reached
        raise AssertionError(f"import_module should not be called for {name}")

    monkeypatch.setattr("braintrust.integrations.base.importlib.import_module", explode)

    assert _import_optional_module("braintrust_fake_sdk") is sentinel


def test_import_optional_module_imports_when_absent():
    assert _import_optional_module("json") is sys.modules["json"]
    assert _import_optional_module("braintrust_module_that_does_not_exist") is None


def test_import_optional_module_waits_for_initializing_module(monkeypatch):
    """A half-built module must not short-circuit the import machinery.

    The loader puts a module in sys.modules *before* running its body, so
    during a concurrent import the attributes a patcher looks for do not
    exist yet. Returning it would make the target look absent and silently
    skip instrumentation, so we must fall through and let import_module
    block on the per-module lock.
    """
    # module_from_spec wires up __spec__ the way the real loader does, so the
    # flag can be flipped on the spec itself.
    spec = importlib.machinery.ModuleSpec("braintrust_partial_sdk", loader=None)
    spec._initializing = True
    partial = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "braintrust_partial_sdk", partial)

    finished = types.ModuleType("braintrust_partial_sdk")
    monkeypatch.setattr(
        "braintrust.integrations.base.importlib.import_module",
        lambda name: finished,
    )

    assert _import_optional_module("braintrust_partial_sdk") is finished

    # Once initialization completes the shortcut applies again.
    spec._initializing = False
    assert _import_optional_module("braintrust_partial_sdk") is partial
