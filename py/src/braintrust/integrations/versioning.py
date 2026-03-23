import importlib.metadata
from typing import Any

from packaging.specifiers import SpecifierSet
from packaging.version import InvalidVersion, Version


def detect_module_version(module: Any, import_names: tuple[str, ...]) -> str | None:
    candidates: list[str] = []

    module_name = getattr(module, "__name__", None)
    if isinstance(module_name, str) and module_name:
        candidates.append(module_name.split(".")[0])

    module_package = getattr(module, "__package__", None)
    if isinstance(module_package, str) and module_package:
        candidates.append(module_package.split(".")[0])

    candidates.extend(name.split(".")[0] for name in import_names)

    seen: set[str] = set()
    for candidate in candidates:
        if candidate in seen:
            continue
        seen.add(candidate)
        try:
            return importlib.metadata.version(candidate)
        except importlib.metadata.PackageNotFoundError:
            continue

    version = getattr(module, "__version__", None)
    return version if isinstance(version, str) else None


def make_specifier(*, min_version: str | None = None, max_version: str | None = None) -> SpecifierSet:
    """Build a :class:`SpecifierSet` from optional min/max bounds."""
    spec = SpecifierSet(prereleases=True)
    if min_version is not None:
        spec &= SpecifierSet(f">={min_version}", prereleases=True)
    if max_version is not None:
        spec &= SpecifierSet(f"<={max_version}", prereleases=True)
    return spec


def version_satisfies(version: str | None, spec: str | SpecifierSet | None) -> bool:
    """Return True if *version* satisfies the PEP 440 *spec*."""
    if spec is None:
        return True
    if version is None:
        return False
    try:
        ss = spec if isinstance(spec, SpecifierSet) else SpecifierSet(spec, prereleases=True)
        return Version(version) in ss
    except InvalidVersion:
        return False
