import json
import subprocess
import sys

import pytest


requires_lazy_imports = pytest.mark.skipif(
    sys.version_info < (3, 15), reason="__lazy_modules__ (PEP 810) is a no-op before Python 3.15"
)


def _run_import_script(script: str) -> dict[str, object]:
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout.splitlines()[-1])


def test_import_braintrust_does_not_load_integrations_or_providers():
    loaded = _run_import_script(
        """
import importlib.abc
import json
import sys

class RejectProviderImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "openai" or fullname.startswith("openai.") or fullname == "anthropic" or fullname.startswith("anthropic."):
            raise AssertionError(f"unexpected provider import: {fullname}")
        return None

sys.meta_path.insert(0, RejectProviderImports())
import braintrust

print(json.dumps({
    "integrations": sorted(name for name in sys.modules if name.startswith("braintrust.integrations")),
    "providers": sorted(name for name in sys.modules if name == "openai" or name.startswith("openai.") or name == "anthropic" or name.startswith("anthropic.")),
}))
"""
    )

    assert loaded == {"integrations": [], "providers": []}


def test_top_level_integration_exports_load_only_when_called():
    loaded = _run_import_script(
        """
import json
import inspect
import sys
import braintrust
from braintrust import setup_ai_sdk, setup_pydantic_ai, wrap_anthropic, wrap_instructor, wrap_litellm, wrap_openai, wrap_openrouter

before = sorted(name for name in sys.modules if name.startswith("braintrust.integrations"))
braintrust.auto_instrument(
    **{name: False for name in inspect.signature(braintrust.auto_instrument).parameters if name != "span_customizers"}
)
after_disabled_auto = sorted(name for name in sys.modules if name.startswith("braintrust.integrations"))

print(json.dumps({"before": before, "after_disabled_auto": after_disabled_auto}))
"""
    )

    assert loaded == {"before": [], "after_disabled_auto": []}


def test_integrations_package_lazily_resolves_public_classes():
    loaded = _run_import_script(
        """
import json
import sys
import braintrust.integrations as integrations

before = sorted(name for name in sys.modules if name.startswith("braintrust.integrations."))
integration = integrations.OpenAIIntegration
after = sorted(name for name in sys.modules if name.startswith("braintrust.integrations."))

print(json.dumps({
    "before": before,
    "resolved_name": integration.__name__,
    "loaded_openai": "braintrust.integrations.openai" in after,
    "loaded_anthropic": "braintrust.integrations.anthropic" in after,
}))
"""
    )

    assert loaded == {
        "before": [],
        "resolved_name": "OpenAIIntegration",
        "loaded_openai": True,
        "loaded_anthropic": False,
    }


@requires_lazy_imports
def test_import_braintrust_defers_lazy_modules():
    loaded = _run_import_script(
        """
import json
import sys

import braintrust

declared = set()
for name, module in list(sys.modules.items()):
    if name.startswith("braintrust"):
        declared.update(getattr(module, "__lazy_modules__", ()))

print(json.dumps({
    "declared": sorted(declared),
    "loaded": sorted(name for name in declared | {"multiprocessing"} if name in sys.modules),
}))
"""
    )

    assert {"asyncio", "jsonschema", "tqdm.auto"} <= set(loaded["declared"])
    assert loaded["loaded"] == []


@requires_lazy_imports
def test_eval_and_parameters_work_with_lazy_modules():
    result = _run_import_script(
        """
import json

from braintrust import Eval
from braintrust.parameters import validate_json_schema


def exact_match(input, output, expected):
    return float(output == expected)


eval_result = Eval(
    "lazy-imports-smoke",
    data=[{"input": 1, "expected": 2}, {"input": 2, "expected": 3}],
    task=lambda input: input + 1,
    scores=[exact_match],
    no_send_logs=True,
)

schema = {"type": "object", "properties": {"n": {"type": "integer", "default": 7}}}
try:
    validate_json_schema({"n": "not-an-int"}, schema)
    invalid_error = None
except ValueError as exc:
    invalid_error = str(exc)

print(json.dumps({
    "outputs": sorted(r.output for r in eval_result.results),
    "scores": sorted(r.scores["exact_match"] for r in eval_result.results),
    "validated": validate_json_schema({}, schema),
    "invalid_error": invalid_error,
}))
"""
    )

    assert result == {
        "outputs": [2, 3],
        "scores": [1.0, 1.0],
        "validated": {"n": 7},
        "invalid_error": "Invalid parameters: n: 'not-an-int' is not of type 'integer'",
    }
