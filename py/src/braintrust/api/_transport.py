"""Legacy HTTP transport primitives used by the Braintrust SDK."""

import sys
import time
from collections.abc import Mapping
from typing import Any

import requests
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..env import BraintrustEnv
from ..util import _urljoin, response_raise_for_status


class RetryRequestExceptionsAdapter(HTTPAdapter):
    """An HTTP adapter that automatically retries requests on connection exceptions.

    This adapter extends requests' HTTPAdapter to add retry logic for common network-related
    exceptions including connection errors, timeouts, and other HTTP errors. It implements
    an exponential backoff strategy between retries to avoid overwhelming servers during
    intermittent connectivity issues.

    Attributes:
        base_num_retries: Maximum number of retries before giving up and re-raising the exception.
        backoff_factor: A multiplier used to determine the time to wait between retries.
                       The actual wait time is calculated as: backoff_factor * (2 ** retry_count).
        default_timeout_secs: Default timeout in seconds for requests that don't specify one.
                             Prevents indefinite hangs on stale connections.
    """

    def __init__(
        self,
        *args: Any,
        base_num_retries: int = 0,
        backoff_factor: float = 0.5,
        default_timeout_secs: float = 60,
        **kwargs: Any,
    ):
        self.base_num_retries = base_num_retries
        self.backoff_factor = backoff_factor
        self.default_timeout_secs = default_timeout_secs
        super().__init__(*args, **kwargs)

    def send(self, *args, **kwargs):
        # Apply default timeout if none provided to prevent indefinite hangs
        if kwargs.get("timeout") is None:
            kwargs["timeout"] = self.default_timeout_secs

        num_prev_retries = 0
        while True:
            try:
                response = super().send(*args, **kwargs)
                # Fully-download the content to ensure we catch any errors from
                # downloading.
                if not response.is_redirect and response.content:
                    pass
                return response
            except (urllib3.exceptions.HTTPError, requests.exceptions.RequestException) as e:
                if num_prev_retries < self.base_num_retries:
                    if isinstance(e, requests.exceptions.ReadTimeout):
                        # Clear all connection pools to discard stale connections. This
                        # fixes hangs caused by NAT gateways silently dropping idle TCP
                        # connections (e.g., Azure's ~4 min timeout). close() calls
                        # PoolManager.clear() which is thread-safe: in-flight requests
                        # keep their checked-out connections, and new requests create
                        # fresh pools on demand.
                        self.close()
                    # Emulates the sleeping logic in the backoff_factor of urllib3 Retry
                    sleep_s = self.backoff_factor * (2**num_prev_retries)
                    print("Retrying request after error:", e, file=sys.stderr)
                    print("Sleeping for", sleep_s, "seconds", file=sys.stderr)
                    time.sleep(sleep_s)
                    num_prev_retries += 1
                else:
                    raise e


class HTTPConnection:
    def __init__(self, base_url: str, adapter: HTTPAdapter | None = None):
        self.base_url = base_url
        self.token = None
        self.adapter = adapter

        self._reset(total=0)

    def ping(self) -> bool:
        try:
            resp = self.get("ping")
            return resp.ok
        except requests.exceptions.ConnectionError:
            return False

    def make_long_lived(self) -> None:
        if not self.adapter:
            timeout_secs = BraintrustEnv.HTTP_TIMEOUT.get(60.0)
            self.adapter = RetryRequestExceptionsAdapter(
                base_num_retries=10, backoff_factor=0.5, default_timeout_secs=timeout_secs
            )
        self._reset()

    @staticmethod
    def sanitize_token(token: str) -> str:
        return token.rstrip("\n")

    def set_token(self, token: str) -> None:
        token = HTTPConnection.sanitize_token(token)
        self.token = token
        self._set_session_token()

    def _set_adapter(self, adapter: HTTPAdapter | None) -> None:
        self.adapter = adapter

    def _reset(self, **retry_kwargs: Any) -> None:
        self.session = requests.Session()

        adapter = self.adapter
        if adapter is None:
            retry = Retry(**retry_kwargs)
            adapter = HTTPAdapter(max_retries=retry)

        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        self._set_session_token()

    def _set_session_token(self) -> None:
        if self.token:
            self.session.headers.update({"Authorization": f"Bearer {self.token}"})

    def get(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.get(_urljoin(self.base_url, path), *args, **kwargs)

    def post(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.post(_urljoin(self.base_url, path), *args, **kwargs)

    def patch(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.patch(_urljoin(self.base_url, path), *args, **kwargs)

    def put(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.put(_urljoin(self.base_url, path), *args, **kwargs)

    def delete(self, path: str, *args: Any, **kwargs: Any) -> requests.Response:
        return self.session.delete(_urljoin(self.base_url, path), *args, **kwargs)

    def get_json(self, object_type: str, args: Mapping[str, Any] | None = None) -> Mapping[str, Any]:
        resp = self.get(f"/{object_type}", params=args)
        response_raise_for_status(resp)
        return resp.json()

    def post_json(self, object_type: str, args: Mapping[str, Any] | None = None) -> Any:
        resp = self.post(f"/{object_type.lstrip('/')}", json=args)
        response_raise_for_status(resp)
        return resp.json()

    def patch_json(self, object_type: str, args: Mapping[str, Any] | None = None) -> Any:
        resp = self.patch(f"/{object_type.lstrip('/')}", json=args)
        response_raise_for_status(resp)
        return resp.json()
