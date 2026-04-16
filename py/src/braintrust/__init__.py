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


if os.getenv("BRAINTRUST_INSTRUMENT_THREADS", "").lower() in ("true", "1", "yes"):
    try:
        from .wrappers.threads import setup_threads

        setup_threads()
    except Exception:
        pass  # Never break on import

from .audit import *
from .auto import auto_instrument as auto_instrument
from .framework import *
from .framework2 import *
from .functions.invoke import *
from .functions.stream import *
from .generated_types import *
from .integrations.anthropic import wrap_anthropic as wrap_anthropic
from .integrations.litellm import wrap_litellm as wrap_litellm
from .integrations.openai import wrap_openai as wrap_openai
from .integrations.openrouter import wrap_openrouter as wrap_openrouter
from .integrations.pydantic_ai import setup_pydantic_ai as setup_pydantic_ai
from .logger import *
from .logger import (
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
