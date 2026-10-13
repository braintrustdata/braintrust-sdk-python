import json
import subprocess
import sys

import pytest


# Modules that `import braintrust` defers through PEP 810 `__lazy_modules__`.
DEFERRED_MODULES = ("asyncio", "gzip", "jsonschema", "multiprocessing", "slugify", "tqdm")


def _run_import_script(script: str) -> dict[str, object]:
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout.splitlines()[-1])


@pytest.mark.skipif(sys.version_info < (3, 15), reason="__lazy_modules__ (PEP 810) is a no-op before Python 3.15")
def test_import_braintrust_defers_heavy_modules():
    loaded = _run_import_script(
        f"""
import json
import sys

import braintrust

print(json.dumps(sorted(name for name in {DEFERRED_MODULES!r} if name in sys.modules)))
"""
    )

    assert loaded == []


def test_eval_and_parameters_work_with_deferred_imports():
    loaded = _run_import_script(
        f"""
import json
import sys

from braintrust import Eval
from braintrust.parameters import validate_json_schema


def exact_match(input, output, expected):
    return float(output == expected)


result = Eval(
    "lazy-imports-smoke",
    data=[{{"input": 1, "expected": 2}}, {{"input": 2, "expected": 3}}],
    task=lambda input: input + 1,
    scores=[exact_match],
    no_send_logs=True,
)

schema = {{"type": "object", "properties": {{"n": {{"type": "integer", "default": 7}}}}}}
validated = validate_json_schema({{}}, schema)
try:
    validate_json_schema({{"n": "not-an-int"}}, schema)
    invalid_error = None
except ValueError as exc:
    invalid_error = str(exc)

print(json.dumps({{
    "outputs": sorted(r.output for r in result.results),
    "scores": sorted(r.scores["exact_match"] for r in result.results),
    "validated": validated,
    "invalid_error": invalid_error,
    "loaded": sorted(name for name in ("asyncio", "jsonschema", "tqdm") if name in sys.modules),
}}))
"""
    )

    assert loaded == {
        "outputs": [2, 3],
        "scores": [1.0, 1.0],
        "validated": {"n": 7},
        "invalid_error": "Invalid parameters: n: 'not-an-int' is not of type 'integer'",
        "loaded": ["asyncio", "jsonschema", "tqdm"],
    }
