import pathlib
import sys

import pyperf


if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from braintrust.bt_json import _to_bt_safe, bt_safe_deep_copy

from benchmarks._utils import disable_pyperf_psutil
from benchmarks.fixtures import make_bt_safe_deep_copy_cases, make_to_bt_safe_cases


def main(runner: pyperf.Runner | None = None) -> None:
    if runner is None:
        disable_pyperf_psutil()
        runner = pyperf.Runner()

    for case_name, value in make_to_bt_safe_cases():
        runner.bench_func(f"bt_json._to_bt_safe[{case_name}]", _to_bt_safe, value)

    for case_name, value in make_bt_safe_deep_copy_cases():
        runner.bench_func(f"bt_json.bt_safe_deep_copy[{case_name}]", bt_safe_deep_copy, value)


if __name__ == "__main__":
    main()
