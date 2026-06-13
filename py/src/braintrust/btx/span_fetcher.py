"""Fetch brainstore spans from the Braintrust API after a real (non-VCR) run.

Mirrors the BTQL query logic in sdk-test's ``btx/utils.py`` and the Java
``SpanFetcher`` class.  Only used in VCR-off mode; in VCR-on mode spans are
captured in memory by the memory background logger.
"""

from __future__ import annotations

import os
import time
from typing import Any

import requests
from braintrust.env import BraintrustEnv


_BACKOFF_SECONDS = 30
_MAX_TOTAL_WAIT_SECONDS = 600


def fetch_project_id(project_name: str) -> str:
    """Resolve a project name to its UUID via the Braintrust REST API."""
    api_key = _require_api_key()
    api_url = _api_url()

    resp = requests.get(
        f"{api_url}/v1/project",
        headers=_auth_headers(api_key),
        params={"project_name": project_name},
        timeout=30,
    )
    resp.raise_for_status()
    projects = resp.json().get("objects", [])
    matching = [p for p in projects if p.get("name") == project_name]
    if not matching:
        raise ValueError(f"Braintrust project not found: {project_name!r}")
    return matching[0]["id"]


def fetch_spans(root_span_id: str, project_id: str, num_expected: int) -> list[dict[str, Any]]:
    """Fetch child spans for *root_span_id* with retry/backoff.

    Retries every ``_BACKOFF_SECONDS`` seconds for up to
    ``_MAX_TOTAL_WAIT_SECONDS`` total wait time.

    Args:
        root_span_id: The root span ID returned by ``logger.start_span()``.
        project_id: Braintrust project UUID.
        num_expected: Number of child (LLM) spans expected.

    Returns:
        List of span dicts sorted by ``created`` ascending, scorer spans
        excluded.
    """
    total_wait = 0
    last_error: Exception | None = None

    while True:
        try:
            return _fetch_once(root_span_id, project_id, num_expected)
        except _RetriableError as exc:
            last_error = exc
            if total_wait >= _MAX_TOTAL_WAIT_SECONDS:
                break
            print(f"[btx] Spans not ready yet, retrying in {_BACKOFF_SECONDS}s (waited {total_wait}s so far): {exc}")
            time.sleep(_BACKOFF_SECONDS)
            total_wait += _BACKOFF_SECONDS
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code in (429, 504):
                last_error = exc
                if total_wait >= _MAX_TOTAL_WAIT_SECONDS:
                    break
                print(f"[btx] HTTP {exc.response.status_code}, retrying in {_BACKOFF_SECONDS}s...")
                time.sleep(_BACKOFF_SECONDS)
                total_wait += _BACKOFF_SECONDS
            else:
                raise

    raise TimeoutError(
        f"Timed out waiting for {num_expected} span(s) after {_MAX_TOTAL_WAIT_SECONDS}s. Last error: {last_error}"
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


class _RetriableError(Exception):
    """Raised when spans aren't ready yet; caller should retry."""


def _fetch_once(root_span_id: str, project_id: str, num_expected: int) -> list[dict[str, Any]]:
    api_key = _require_api_key()
    api_url = _api_url()

    btql_query = {
        "query": {
            "select": [{"op": "star"}],
            "from": {
                "op": "function",
                "name": {"op": "ident", "name": ["project_logs"]},
                "args": [{"op": "literal", "value": project_id}],
            },
            "filter": {
                "op": "and",
                "left": {
                    "op": "eq",
                    "left": {"op": "ident", "name": ["root_span_id"]},
                    "right": {"op": "literal", "value": root_span_id},
                },
                "right": {
                    "op": "ne",
                    "left": {"op": "ident", "name": ["span_parents"]},
                    "right": {"op": "literal", "value": None},
                },
            },
            "sort": [{"expr": {"op": "ident", "name": ["created"]}, "dir": "asc"}],
            "limit": 1000,
        },
        "use_columnstore": True,
        "use_brainstore": True,
        "brainstore_realtime": True,
    }

    resp = requests.post(
        f"{api_url}/btql",
        headers=_auth_headers(api_key),
        json=btql_query,
        timeout=30,
    )
    resp.raise_for_status()

    all_spans: list[dict] = resp.json().get("data", [])

    # Strip scorer spans injected by the Braintrust backend
    spans = [s for s in all_spans if (s.get("span_attributes") or {}).get("purpose") != "scorer"]

    if len(spans) == 0:
        raise _RetriableError(f"No child spans found yet for root_span_id={root_span_id!r}")

    if len(spans) < num_expected:
        raise _RetriableError(f"Found {len(spans)}/{num_expected} child spans for root_span_id={root_span_id!r}")

    if len(spans) > num_expected:
        raise RuntimeError(
            f"Expected {num_expected} child spans but found {len(spans)} — too many (non-retriable). "
            f"root_span_id={root_span_id!r}"
        )

    # Retry if a span arrived but its payload hasn't been indexed yet
    for span in spans:
        if span.get("output") is None and span.get("metrics") is None:
            raise _RetriableError(f"Span arrived but output/metrics not yet indexed (span_id={span.get('span_id')!r})")

    return spans


def _require_api_key() -> str:
    key = BraintrustEnv.API_KEY.get(None, use_dotenv=True)
    if not key:
        raise ValueError("BRAINTRUST_API_KEY is not set in the environment or nearest .env.braintrust file")
    return key


def _api_url() -> str:
    return os.environ.get("BRAINTRUST_API_URL", "https://api.braintrust.dev").rstrip("/")


def _auth_headers(api_key: str) -> dict[str, str]:
    return {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
