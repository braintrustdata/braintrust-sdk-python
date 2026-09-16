"""Both import orders and opt-out, using the real REST ranking cassette."""

import inspect
import subprocess
import sys
from pathlib import Path
from urllib.parse import urlsplit

import yaml
from braintrust.auto import auto_instrument
from braintrust.integrations.conftest import _versioned_cassette_dir
from braintrust.integrations.test_utils import autoinstrument_test_context


if len(sys.argv) == 1:
    for order in ("before", "after"):
        subprocess.run([sys.executable, __file__, order], check=True)
    print("SUCCESS")
    sys.exit(0)

options = {name: False for name in inspect.signature(auto_instrument).parameters}
assert auto_instrument(**options) == {}
RankServiceClient = None
if sys.argv[1] == "before":
    from google.cloud.discoveryengine_v1 import RankServiceClient

options["google_discoveryengine"] = True
assert auto_instrument(**options) == {"google_discoveryengine": True}
assert auto_instrument(**options) == {"google_discoveryengine": True}
from google.auth.credentials import AnonymousCredentials


if sys.argv[1] == "after":
    from google.cloud.discoveryengine_v1 import RankServiceClient


cassette_dir = Path(
    _versioned_cassette_dir(str(Path(__file__).parent.parent / "google_discoveryengine" / "cassettes"))
)
cassette = yaml.safe_load((cassette_dir / "test_rank.yaml").read_text())
ranking_config = urlsplit(cassette["interactions"][0]["request"]["uri"]).path.removeprefix("/v1/").split(":rank")[0]

assert RankServiceClient is not None
with autoinstrument_test_context(
    "test_rank", integration="google_discoveryengine", vcr_config={"record_mode": "none"}
) as memory_logger:
    client = RankServiceClient(transport="rest", credentials=AnonymousCredentials())
    result = client.rank(
        request={
            "ranking_config": ranking_config,
            "model": "semantic-ranker-512@latest",
            "query": "What is Braintrust?",
            "records": [
                {"id": "1", "content": "Braintrust is a platform for evaluating and monitoring AI applications."},
                {"id": "2", "content": "The moon orbits the Earth."},
            ],
            "top_n": 1,
        },
        retry=None,
    )
    assert result.records[0].id == "1"
    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["metadata"]["provider"] == "google"
    assert spans[0]["context"]["span_origin"]["instrumentation"]["name"] == "google-discoveryengine-auto"
