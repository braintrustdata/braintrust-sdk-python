"""Shared helpers for benchmark modules."""

import sys


def disable_pyperf_psutil() -> None:
    """Disable pyperf's psutil-based metadata collection on macOS.

    pyperf's optional system metadata collection can hit sandboxed sysctl
    paths on macOS.  Disabling it keeps local runs portable.
    """
    if sys.platform != "darwin":
        return

    import pyperf._collect_metadata as collect_metadata
    import pyperf._cpu_utils as cpu_utils

    collect_metadata.psutil = None
    cpu_utils.psutil = None
