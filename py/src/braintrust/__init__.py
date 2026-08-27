"""
A Python library for interacting with [Braintrust](https://braintrust.dev/). This library
contains functionality for running evaluations, logging completions, loading and invoking
functions, and more.

`braintrust` is distributed as a [library on PyPI](https://pypi.org/project/braintrust/). It is open source and
[available on GitHub](https://github.com/braintrustdata/braintrust-sdk-python/tree/main/py).

### Quickstart

Install the library with pip.

```bash
pip install braintrust
```

Then, create a file like `eval_hello.py` with the following content:

```python
from braintrust import Eval

def is_equal(expected, output):
    return expected == output

Eval(
  "Say Hi Bot",
  data=lambda: [
      {
          "input": "Foo",
          "expected": "Hi Foo",
      },
      {
          "input": "Bar",
          "expected": "Hello Bar",
      },
  ],  # Replace with your eval dataset
  task=lambda input: "Hi " + input,  # Replace with your LLM call
  scores=[is_equal],
)
```

Finally, run the script with `braintrust eval eval_hello.py`.

```bash
BRAINTRUST_API_KEY=<YOUR_BRAINTRUST_API_KEY> braintrust eval eval_hello.py
```

### API Reference
"""

# Check env var at import time for auto-instrumentation
import os
from typing import TYPE_CHECKING


if TYPE_CHECKING:
    # These names must precede the generated wildcard so type checkers keep the
    # runtime resource classes for the four names shared by both modules.
    from .logger import Dataset as Dataset  # noqa: I001
    from .logger import Experiment as Experiment
    from .logger import Project as Project
    from .logger import Prompt as Prompt
    from .generated_types import *


if os.getenv("BRAINTRUST_INSTRUMENT_THREADS", "").lower() in ("true", "1", "yes"):
    try:
        from .wrappers.threads import setup_threads

        setup_threads()
    except Exception:
        pass  # Never break on import

from . import generated_types as _generated_types  # noqa: I001
from .audit import *
from .auto import auto_instrument as auto_instrument
from .dataset_pipeline import *
from .framework import *
from .framework2 import *
from .functions.invoke import *
from .functions.stream import *

# Keep this before the logger wildcard so its existing runtime collision
# precedence remains unchanged while new generated names are picked up.
for _name in _generated_types.__all__:
    if _name not in {"Dataset", "Experiment", "Project", "Prompt"}:
        globals()[_name] = getattr(_generated_types, _name)

from .integrations.ai_sdk import setup_ai_sdk as setup_ai_sdk  # noqa: I001
from .integrations.anthropic import wrap_anthropic as wrap_anthropic
from .integrations.instructor import wrap_instructor as wrap_instructor
from .integrations.litellm import wrap_litellm as wrap_litellm
from .integrations.openai import wrap_openai as wrap_openai
from .integrations.openrouter import wrap_openrouter as wrap_openrouter
from .integrations.pydantic_ai import setup_pydantic_ai as setup_pydantic_ai
from .logger import *
from .logger import (
    Dataset as Dataset,
    Experiment as Experiment,
    Project as Project,
    Prompt as Prompt,
    _internal_get_global_state,  # noqa: F401 # type: ignore[reportUnusedImport]
    _internal_reset_global_state,  # noqa: F401 # type: ignore[reportUnusedImport]
    _internal_with_custom_background_logger,  # noqa: F401 # type: ignore[reportUnusedImport]
)
from .sandbox import RegisteredSandboxFunction as RegisteredSandboxFunction
from .sandbox import RegisterSandboxResult as RegisterSandboxResult
from .sandbox import SandboxConfig as SandboxConfig
from .sandbox import register_sandbox as register_sandbox
from .util import BT_IS_ASYNC_ATTRIBUTE as BT_IS_ASYNC_ATTRIBUTE
from .util import MarkAsyncWrapper as MarkAsyncWrapper
