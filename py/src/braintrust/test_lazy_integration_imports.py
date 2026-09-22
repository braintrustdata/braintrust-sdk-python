import json
import subprocess
import sys


def _run_import_script(script: str) -> dict[str, object]:
    result = subprocess.run([sys.executable, "-c", script], check=True, capture_output=True, text=True)
    return json.loads(result.stdout)


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
