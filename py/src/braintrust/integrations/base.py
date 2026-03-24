"""Shared integration orchestration primitives."""

import importlib
import inspect
import re
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


class FunctionWrapperPatcher(BasePatcher):
    """Base patcher for single-target `wrap_function_wrapper` instrumentation.

    Set ``target_module`` to an import path when the patch target lives in a
    different module than the one provided by the integration (e.g. a deep
    submodule that may or may not be installed).  The module is imported lazily
    when the patcher is evaluated.
    """

    target_path: ClassVar[str]
    wrapper: ClassVar[Any]
    target_module: ClassVar[str | None] = None

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
        """Return whether the target exists and this patcher's version gate passes."""
        return (
            super().applies(module, version, target=target)
            and cls.resolve_target(module, version, target=target) is not None
        )

    @classmethod
    def patch_marker_attr(cls) -> str:
        """Return the sentinel attribute used to mark this target as patched."""
        suffix = re.sub(r"\W+", "_", cls.name).strip("_")
        return f"__braintrust_patched_{suffix}__"

    @classmethod
    def mark_patched(cls, obj: Any) -> None:
        """Mark a wrapped target so future patch attempts are idempotent."""
        setattr(obj, cls.patch_marker_attr(), True)

    @classmethod
    def is_patched(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Return whether this patcher's target has already been instrumented."""
        resolved_target = cls.resolve_target(module, version, target=target)
        return bool(resolved_target is not None and getattr(resolved_target, cls.patch_marker_attr(), False))

    @classmethod
    def patch(cls, module: Any | None, version: str | None, *, target: Any | None = None) -> bool:
        """Apply instrumentation for this patcher."""
        root = cls.resolve_root(module, version, target=target)
        if root is None or not cls.applies(module, version, target=target):
            return False

        wrap_function_wrapper(root, cls.target_path, cls.wrapper)
        resolved_target = cls.resolve_target(module, version, target=target)
        if resolved_target is None:
            return False

        cls.mark_patched(resolved_target)
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
        *target* or the patch has already been applied.  Returns *target*
        for convenient chaining.
        """
        marker = cls.patch_marker_attr()
        if getattr(target, marker, False):
            return target
        attr = cls.target_path.rsplit(".", 1)[-1]
        if _resolve_attr_path(target, attr) is None:
            return target
        wrap_function_wrapper(target, attr, cls.wrapper)
        cls.mark_patched(target)
        return target


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
            if patcher.is_patched(module, version, target=target):
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
