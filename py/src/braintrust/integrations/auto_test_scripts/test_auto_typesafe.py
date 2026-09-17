"""Test auto_instrument for TypeSafe."""

import os

from braintrust.auto import auto_instrument
from braintrust.integrations.test_utils import autoinstrument_test_context
from typesafe_sdk import Noul, TypeSafeClient


results = auto_instrument()
assert results.get("typesafe") is True
assert auto_instrument().get("typesafe") is True

with autoinstrument_test_context("test_auto_typesafe", integration="typesafe") as memory_logger:
    with TypeSafeClient(api_key=os.environ["TYPESAFE_API_KEY"]) as client:
        response = client.system_one(
            state="The package arrived intact and on time.",
            questions={"positive": Noul(instructions="Is this feedback positive?")},
        )
    assert 0 <= response.nouls["positive"].noul <= 1

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["provider"] == "typesafe"
    assert span["metadata"]["model"].startswith("jev-")
    assert span["input"]["questions"][0]["id"] == "positive"
    assert span["output"]["answers"][0]["id"] == "positive"

print("SUCCESS")
