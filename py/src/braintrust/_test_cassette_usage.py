"""Record which cassette files a test process reads.

Used by ``py/scripts/check-unused-cassettes.py`` to find cassettes that no test
loads. When ``BRAINTRUST_CASSETTE_USAGE_DIR`` is set, :func:`install` registers
an audit hook that logs every cassette file opened for reading under the
``braintrust`` package. This catches every loader (pytest-recording, the Claude
Agent SDK transport, the gRPC recorder, btx specs, ...) because they all go
through ``open()``.

Each process writes its own ``usage-<pid>-<random>.txt`` file in that directory,
one path per line, relative to the ``braintrust`` package directory and
POSIX-style so logs from different runners (including Windows) can be merged.
Auto-instrument test subprocesses inherit the env var and install the hook
through :mod:`braintrust.integrations.test_utils`.
"""

import os
import sys
import uuid


USAGE_DIR_ENV = "BRAINTRUST_CASSETTE_USAGE_DIR"

# Auto-instrument subprocesses may import braintrust from site-packages while
# reading cassettes from the source tree handed down via this env var.
_PACKAGE_DIR = (
    os.path.dirname(os.path.abspath(os.environ["BRAINTRUST_INTEGRATIONS_DIR"]))
    if os.environ.get("BRAINTRUST_INTEGRATIONS_DIR")
    else os.path.dirname(os.path.abspath(__file__))
)
_CASSETTES_SEGMENT = f"{os.sep}cassettes{os.sep}"
_WRITE_FLAGS = os.O_WRONLY | os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_TRUNC

_installed = False


def _is_read(mode, flags) -> bool:
    if isinstance(mode, str):
        return not any(c in mode for c in "wax+")
    # os.open() reports mode=None and passes the flags instead.
    return not (isinstance(flags, int) and flags & _WRITE_FLAGS)


def install() -> bool:
    """Start recording cassette reads if ``BRAINTRUST_CASSETTE_USAGE_DIR`` is set.

    Safe to call more than once. Returns True when recording is active.
    """
    global _installed
    if _installed:
        return True
    usage_dir = os.environ.get(USAGE_DIR_ENV)
    if not usage_dir:
        return False

    os.makedirs(usage_dir, exist_ok=True)
    log_path = os.path.join(usage_dir, f"usage-{os.getpid()}-{uuid.uuid4().hex[:8]}.txt")
    # Open the log once up front: writes to an already-open file don't raise
    # "open" audit events, so the hook can't recurse into itself.
    log = open(log_path, "a", encoding="utf-8")
    seen: set[str] = set()

    def hook(event, args):
        if event != "open":
            return
        try:
            path, mode, flags = args
            if isinstance(path, int) or not _is_read(mode, flags):
                return
            path = os.path.abspath(os.fsdecode(path))
            if path in seen or _CASSETTES_SEGMENT not in path or not path.startswith(_PACKAGE_DIR):
                return
            seen.add(path)
            log.write(os.path.relpath(path, _PACKAGE_DIR).replace(os.sep, "/") + "\n")
            log.flush()
        except Exception:  # pylint: disable=broad-except
            # Audit hooks must never break the code under test.
            pass

    sys.addaudithook(hook)
    _installed = True
    return True
