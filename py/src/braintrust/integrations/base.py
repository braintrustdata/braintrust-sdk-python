"""Shared integration orchestration primitives."""

import importlib
import inspect
import re
import sys
from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, ClassVar

from wrapt import wrap_function_wrapper

from .versioning import detect_module_version, make_specifier, version_satisfies


class BasePatcher(ABC):
    """Base class for one concrete integration patch strategy."""

    name: ClassVar[str]
    patch_id: ClassVar[str | None] = None
    version_spec: ClassVar[str | None] = None
    priority: ClassVar[int] = 100
    rescan_on_setup: ClassVar[bool] = False

    @classmethod
    def patch_marker_attr(cls) -> str:
        """Return the sentinel attribute used to mark this patcher as applied."""
        suffix = re.sub(r"\W+", "_", cls.identifier()).strip("_")
        return f"__braintrust_patched_{suffix}__"

    @classmethod
    def has_patch_marker(cls, obj: Any) -> bool:
        """Return whether *obj* is marked as patched by this patcher.

        For classes, read ``__dict__`` directly so markers inherited via the
        MRO do not make subclasses appear locally patched.
        """
        if obj is None:
            return False
        if isinstance(obj, type):
            return bool(obj.__dict__.get(cls.patch_marker_attr(), False))
        return bool(getattr(obj, cls.patch_marker_attr(), False))

    @classmethod
    def mark_patched(cls, obj: Any) -> None:
        """Mark an object as patched by this patcher."""
        setattr(obj, cls.patch_marker_attr(), True)

    @classmethod
    def identifier(cls) -> str:
        """Return the public identifier for selecting this patcher."""
        return cls.patch_id or cls.name

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether this patcher should run for the given module/version."""
        return version_satisfies(version, cls.version_spec)

    @classmethod
    @abstractmethod
    def is_patched(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether this patcher's target has already been instrumented."""
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def patch(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Apply instrumentation for this patcher."""
        raise NotImplementedError


class CallbackPatcher(BasePatcher):
    """Base patcher for integration setup steps that only execute a callback.

    Use this for integrations that do not need in-place wrapping or class
    replacement, but still need an idempotent setup action once a target module
    is available — for example, registering a global callback handler.

    Set ``target_module`` when the callback should only run if a particular
    optional module can be imported. Provide ``state_getter`` when patch state
    should be derived from integration-owned state instead of a marker stored on
    the resolved root object.
    """

    target_module: ClassVar[str | None] = None
    callback: ClassVar[Any]
    state_getter: ClassVar[Any | None] = None

    @classmethod
    def resolve_root(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> Any | None:
        """Return the object whose availability gates this callback patcher."""
        if target is not None:
            return target
        if cls.target_module is not None:
            try:
                return importlib.import_module(cls.target_module)
            except ImportError:
                return None
        return module

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether the callback should run for the given module/version."""
        return (
            super().applies(module, version, target=target)
            and cls.resolve_root(module, version, target=target) is not None
        )

    @classmethod
    def is_patched(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether this callback patcher has already been applied."""
        if cls.state_getter is not None:
            return bool(cls.state_getter())
        root = cls.resolve_root(module, version, target=target)
        return bool(root is not None and cls.has_patch_marker(root))

    @classmethod
    def patch(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Execute the callback and mark the root as patched when needed."""
        root = cls.resolve_root(module, version, target=target)
        if root is None or not cls.applies(module, version, target=target):
            return False

        result = cls.callback()
        if result is False:
            return False

        if cls.state_getter is not None:
            return cls.is_patched(module, version, target=target)

        cls.mark_patched(root)
        return True


class ClassScanPatcher(BasePatcher):
    """Base patcher for rescanning and patching discovered class hierarchies."""

    rescan_on_setup: ClassVar[bool] = True
    include_abstract_classes: ClassVar[bool] = False
    target_module: ClassVar[str | None] = None
    root_class_path: ClassVar[str | None] = None

    @classmethod
    def resolve_scan_root(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> Any | None:
        """Return the object from which this patcher resolves its root class."""
        if target is not None:
            return target
        if cls.target_module is not None:
            try:
                return importlib.import_module(cls.target_module)
            except ImportError:
                return None
        return module

    @classmethod
    def iter_root_classes(
        cls,
        module: Any | None,
        version: str | None,
        *,
        target: Any | None = None,
    ) -> Iterable[type[Any]]:
        """Yield root classes whose subclass trees should be scanned."""
        if cls.root_class_path is None:
            return ()
        root = cls.resolve_scan_root(module, version, target=target)
        if root is None:
            return ()
        root_class = _resolve_attr_path(root, cls.root_class_path)
        if root_class is None:
            return ()
        return (root_class,)

    @classmethod
    def resolve_root_classes(
        cls,
        module: Any | None,
        version: str | None,
        *,
        target: Any | None = None,
    ) -> tuple[type[Any], ...]:
        """Return the currently discoverable root classes for this patcher."""
        try:
            return tuple(cls.iter_root_classes(module, version, target=target))
        except ImportError:
            return ()

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether any root classes are currently discoverable."""
        return super().applies(module, version, target=target) and bool(
            cls.resolve_root_classes(module, version, target=target)
        )

    @classmethod
    @abstractmethod
    def patch_class(cls, target_class: type[Any]) -> bool | None:
        """Patch one discovered class.

        Return ``False`` to skip marking the class as patched. Any other return
        value is treated as a successful patch.
        """
        raise NotImplementedError

    @classmethod
    def iter_classes(
        cls,
        module: Any | None,
        version: str | None,
        *,
        target: Any | None = None,
    ) -> Iterable[type[Any]]:
        """Yield discovered subclasses under the configured root classes."""

        def walk(base_class: type[Any]) -> Iterable[type[Any]]:
            for subclass in base_class.__subclasses__():
                if cls.include_abstract_classes or not getattr(subclass, "__abstractmethods__", None):
                    yield subclass
                yield from walk(subclass)

        for root_class in cls.resolve_root_classes(module, version, target=target):
            yield from walk(root_class)

    @classmethod
    def is_patched(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return ``True`` when every currently discovered class is patched."""
        classes = tuple(cls.iter_classes(module, version, target=target))
        return bool(classes) and all(cls.has_patch_marker(class_) for class_ in classes)

    @classmethod
    def patch(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Patch all newly discovered classes under the configured roots."""
        success = False
        for class_ in cls.iter_classes(module, version, target=target):
            if cls.has_patch_marker(class_):
                continue
            if cls.patch_class(class_) is False:
                continue
            cls.mark_patched(class_)
            success = True
        return success


class FunctionWrapperPatcher(BasePatcher):
    """Base patcher for single-target `wrap_function_wrapper` instrumentation.

    Set ``target_module`` to an import path when the patch target lives in a
    different module than the one provided by the integration (e.g. a deep
    submodule that may or may not be installed).  The module is imported lazily
    when the patcher is evaluated.

    Set ``superseded_by`` to a tuple of other ``FunctionWrapperPatcher``
    subclasses that take priority over this patcher.  If any of them apply
    (i.e. their target exists), this patcher yields — both in the
    ``setup()`` path (via ``applies()``) and in the manual ``wrap_target()``
    path.  This is useful for version-conditional mutual exclusion, e.g.
    wrapping a public ``run()`` only when the private ``_run()`` is absent.
    """

    target_path: ClassVar[str]
    wrapper: ClassVar[Any]
    target_module: ClassVar[str | None] = None
    superseded_by: ClassVar[tuple[type["FunctionWrapperPatcher"], ...]] = ()

    @classmethod
    def resolve_root(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> Any | None:
        """Return the root object from which this patcher resolves its target.

        When ``target_module`` is set, the patcher imports that module and uses
        it as root instead of the integration-level module.  If the import
        fails, ``None`` is returned so that ``applies()`` returns ``False``.
        """
        if target is not None:
            return target
        if cls.target_module is not None:
            try:
                return importlib.import_module(cls.target_module)
            except ImportError:
                return None
        return module

    @classmethod
    def resolve_target(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> Any | None:
        """Return the concrete callable or descriptor that this patcher instruments."""
        root = cls.resolve_root(module, version, target=target)
        if root is None:
            return None
        return _resolve_attr_path(root, cls.target_path)

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether the target exists and this patcher's version gate passes.

        Returns ``False`` if any patcher listed in ``superseded_by`` applies.
        """
        if not super().applies(module, version, target=target):
            return False
        if cls.resolve_target(module, version, target=target) is None:
            return False
        for superior in cls.superseded_by:
            if superior.applies(module, version, target=target):
                return False
        return True

    @classmethod
    def patch_marker_attr(cls) -> str:
        """Return the sentinel attribute used to mark this target as patched."""
        suffix = re.sub(r"\W+", "_", cls.name).strip("_")
        return f"__braintrust_patched_{suffix}__"

    @classmethod
    def mark_patched(cls, obj: Any) -> None:
        """Mark a wrapped target so future patch attempts are idempotent."""
        try:
            setattr(obj, cls.patch_marker_attr(), True)
        except AttributeError:
            # Some objects (e.g. bound methods) don't support setattr.
            # Callers that need a fallback location (like ``patch()``) handle
            # this by catching the failure and storing the marker elsewhere.
            pass

    @classmethod
    def is_patched(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether this patcher's target has already been instrumented."""
        resolved_target = cls.resolve_target(module, version, target=target)
        if cls.has_patch_marker(resolved_target):
            return True
        # Fall back to checking the root — the marker may live there when the
        # resolved target does not support setattr (e.g. bound methods).
        root = cls.resolve_root(module, version, target=target)
        if root is not None and root is not resolved_target and cls.has_patch_marker(root):
            return True
        return False

    @classmethod
    def patch(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Apply instrumentation for this patcher."""
        root = cls.resolve_root(module, version, target=target)
        if root is None or not cls.applies(module, version, target=target):
            return False

        wrap_function_wrapper(root, cls.target_path, cls.wrapper)
        resolved_target = _resolve_attr_path(root, cls.target_path)
        if resolved_target is None:
            return False

        marker = cls.patch_marker_attr()
        cls.mark_patched(resolved_target)
        # If mark_patched could not store the marker on the target (e.g. bound
        # methods), store it on the root so is_patched() can still find it.
        if not cls.has_patch_marker(resolved_target):
            setattr(root, marker, True)
        return True

    @classmethod
    def wrap_target(cls, target: Any) -> Any:
        """Patch *target* directly for tracing (idempotent).

        Unlike ``patch()``, which resolves the full ``target_path`` from a
        module root, this method wraps the **leaf** attribute of
        ``target_path`` directly on *target*.  This is useful for manual
        wrapping of a specific class or object (e.g. ``wrap_agent(MyAgent)``).

        The patch marker is set on *target* itself so that callers can check
        ``getattr(target, patcher.patch_marker_attr(), False)`` to detect
        whether the patch has already been applied.

        Returns *target* unchanged if the leaf attribute does not exist on
        *target*, the patch has already been applied, or a patcher in
        ``superseded_by`` has a target that exists on *target*.  Returns
        *target* for convenient chaining.
        """
        if cls.has_patch_marker(target):
            return target
        attr = cls.target_path.rsplit(".", 1)[-1]
        if _resolve_attr_path(target, attr) is None:
            return target
        # Check superseded_by against the target object directly.
        for superior in cls.superseded_by:
            superior_attr = superior.target_path.rsplit(".", 1)[-1]
            if _resolve_attr_path(target, superior_attr) is not None:
                return target
        wrap_function_wrapper(target, attr, cls.wrapper)
        cls.mark_patched(target)
        return target


class ClassReplacementPatcher(BasePatcher):
    """Base patcher for replacing an exported class with a tracing wrapper class.

    Use this when instrumentation cannot be expressed as wrapping one stable
    function or method in place. Typical cases are integrations that need to:

    - replace constructor behavior before the SDK stores callbacks or handlers
    - preserve per-instance state across multiple methods
    - keep ``from provider import Client`` aliases working after setup by
      propagating the replacement to modules that already imported the class

    Prefer ``FunctionWrapperPatcher`` when a stable attribute can be instrumented
    in place with ``wrap_function_wrapper(...)`` and class identity does not need
    to change.
    """

    target_attr: ClassVar[str]
    propagate_imported_aliases: ClassVar[bool] = True
    # Factory that takes the original exported class and returns the replacement class.
    replacement_factory: ClassVar[Any]

    @classmethod
    def resolve_target(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> Any | None:
        """Return the exported class object that this patcher replaces."""
        root = target if target is not None else module
        if root is None:
            return None
        return getattr(root, cls.target_attr, None)

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether the target class exists and the version gate passes."""
        return super().applies(module, version, target=target) and (
            cls.resolve_target(module, version, target=target) is not None
        )

    @classmethod
    def patch_marker_attr(cls) -> str:
        """Return the sentinel attribute used to mark the replacement class as patched."""
        suffix = re.sub(r"\W+", "_", cls.name).strip("_")
        return f"__braintrust_patched_{suffix}__"

    @classmethod
    def mark_patched(cls, obj: Any) -> None:
        """Mark a replacement class so future patch attempts are idempotent."""
        setattr(obj, cls.patch_marker_attr(), True)

    @classmethod
    def is_patched(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether this patcher's replacement class is already installed."""
        resolved_target = cls.resolve_target(module, version, target=target)
        return bool(resolved_target is not None and cls.has_patch_marker(resolved_target))

    @classmethod
    def patch(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Replace the exported class and optionally propagate the new binding."""
        root = target if target is not None else module
        if root is None or not cls.applies(module, version, target=target):
            return False

        original_class = cls.resolve_target(module, version, target=target)
        if original_class is None:
            return False

        replacement_class = cls.replacement_factory(original_class)
        cls.mark_patched(replacement_class)
        setattr(root, cls.target_attr, replacement_class)

        if cls.propagate_imported_aliases and target is None:
            for mod in list(sys.modules.values()):
                if mod is None or not hasattr(mod, cls.target_attr):
                    continue
                if getattr(mod, cls.target_attr, None) is original_class:
                    setattr(mod, cls.target_attr, replacement_class)

        return True


class CompositeFunctionWrapperPatcher(BasePatcher):
    """Patcher that applies multiple ``FunctionWrapperPatcher`` sub-patchers as one unit.

    Use this when several closely related targets should be patched together
    under a single patcher name — for example, patching both the sync and async
    variants of the same method on one class.

    Subclasses declare ``sub_patchers`` as a tuple of ``FunctionWrapperPatcher``
    classes.  The composite delegates ``applies``, ``is_patched``, and ``patch``
    to the sub-patchers, and the composite is considered patched when **all**
    applicable sub-patchers have been applied.
    """

    sub_patchers: ClassVar[tuple[type[FunctionWrapperPatcher], ...]]

    @classmethod
    def applies(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return ``True`` if the version gate passes and at least one sub-patcher applies."""
        if not super().applies(module, version, target=target):
            return False
        return any(sub.applies(module, version, target=target) for sub in cls.sub_patchers)

    @classmethod
    def is_patched(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return ``True`` when every applicable sub-patcher has been applied."""
        applicable = [sub for sub in cls.sub_patchers if sub.applies(module, version, target=target)]
        if not applicable:
            return False
        return all(sub.is_patched(module, version, target=target) for sub in applicable)

    @classmethod
    def patch(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Apply all applicable sub-patchers."""
        success = False
        for sub in cls.sub_patchers:
            if not sub.applies(module, version, target=target):
                continue
            if sub.is_patched(module, version, target=target):
                success = True
                continue
            success = sub.patch(module, version, target=target) or success
        return success

    @classmethod
    def wrap_target(cls, target: Any) -> Any:
        """Patch *target* directly for tracing (idempotent).

        Delegates to each sub-patcher's ``wrap_target``, which individually
        skips sub-patchers whose leaf attribute does not exist on *target*.
        Returns *target* for convenient chaining.
        """
        for sub in cls.sub_patchers:
            sub.wrap_target(target)
        return target


class BaseIntegration(ABC):
    """Base class for an instrumentable third-party integration."""

    name: ClassVar[str]
    import_names: ClassVar[tuple[str, ...]]
    patchers: ClassVar[tuple[type[BasePatcher], ...]] = ()
    min_version: ClassVar[str | None] = None
    max_version: ClassVar[str | None] = None

    @classmethod
    def available_patchers(cls) -> tuple[str, ...]:
        """Return patcher identifiers in declaration order."""
        return tuple(patcher.identifier() for patcher in cls.patchers)

    @classmethod
    def resolve_patchers(cls) -> tuple[type[BasePatcher], ...]:
        """Return all patchers after validating there are no duplicate identifiers."""
        patchers_by_id: dict[str, type[BasePatcher]] = {}
        for patcher in cls.patchers:
            patcher_id = patcher.identifier()
            existing = patchers_by_id.get(patcher_id)
            if existing is not None and existing is not patcher:
                raise ValueError(f"Duplicate patcher identifier {patcher_id!r} for integration {cls.name!r}")
            patchers_by_id[patcher_id] = patcher

        return cls.patchers

    @classmethod
    def setup(
        cls,
        *,
        target: Any | None = None,
    ) -> bool:
        """Apply all applicable patchers for this integration."""
        module = _import_first_available(cls.import_names)
        if module is None:
            return False
        version = detect_module_version(module, cls.import_names)
        if not version_satisfies(version, make_specifier(min_version=cls.min_version, max_version=cls.max_version)):
            return False

        success = False
        selected_patchers = cls.resolve_patchers()
        for patcher in sorted(selected_patchers, key=lambda patcher: patcher.priority):
            if not patcher.applies(module, version, target=target):
                continue
            if not patcher.rescan_on_setup and patcher.is_patched(module, version, target=target):
                success = True
                continue
            success = patcher.patch(module, version, target=target) or success

        return success


def _import_first_available(import_names: Iterable[str]) -> Any | None:
    """Import and return the first available module from the given names."""
    for import_name in import_names:
        try:
            return importlib.import_module(import_name)
        except ImportError:
            continue
    return None


def _resolve_attr_path(root: Any, path: str) -> Any | None:
    current = root
    for part in path.split("."):
        try:
            current = inspect.getattr_static(current, part)
        except AttributeError:
            return None
    return current
