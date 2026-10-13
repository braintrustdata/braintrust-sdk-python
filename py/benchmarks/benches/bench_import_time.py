import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from benchmarks._utils import disable_pyperf_psutil


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    runner.bench_command("import_time[python_startup]", [sys.executable, "-c", "pass"])
    runner.bench_command("import_time[braintrust]", [sys.executable, "-c", "import braintrust"])


if __name__ == "__main__":
    main()
