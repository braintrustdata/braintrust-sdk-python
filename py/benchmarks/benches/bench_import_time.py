import importlib.util
import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks._utils import disable_pyperf_psutil


def _braintrust_src_dir() -> str:
    # Resolve without importing braintrust so each child process does the
    # import cold, from the same checkout the runner would import.
    spec = importlib.util.find_spec("braintrust")
    if spec is None or spec.origin is None:
        raise RuntimeError("braintrust is not importable")
    return str(pathlib.Path(spec.origin).resolve().parents[1])


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    src_dir = _braintrust_src_dir()
    runner.bench_command("import_time[python_startup]", [sys.executable, "-c", "pass"])
    runner.bench_command(
        "import_time[braintrust]",
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {src_dir!r}); import braintrust"],
    )


if __name__ == "__main__":
    main()
