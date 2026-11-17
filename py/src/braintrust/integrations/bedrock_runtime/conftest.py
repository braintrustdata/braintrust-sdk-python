"""Shared fixtures for sync and async Bedrock clients."""

import os

import pytest
from braintrust import logger
from braintrust.test_helpers import init_test_logger


@pytest.fixture(scope="module")
def vcr_config():
    record_mode = "none" if (os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS")) else "once"
    return {
        "record_mode": record_mode,
        "decode_compressed_response": True,
        "filter_headers": [
            "authorization",
            "Authorization",
            "x-amz-security-token",
            "X-Amz-Security-Token",
            "x-amz-date",
            "X-Amz-Date",
            "x-amz-content-sha256",
            "X-Amz-Content-Sha256",
            "amz-sdk-invocation-id",
            "amz-sdk-request",
            "x-amzn-bedrock-api-key",
        ],
    }


@pytest.fixture
def memory_logger():
    init_test_logger("test-boto3-bedrock")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture(autouse=True)
def bedrock_credentials(request, monkeypatch):
    if "AWS_EC2_METADATA_DISABLED" not in os.environ:
        monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")

    # Older botocore cannot authenticate using AWS_BEARER_TOKEN_BEDROCK.
    # Playback needs no live credentials, including in subprocess checks.
    replay_only = (
        request.config.getoption("--vcr-record", default=None) == "none"
        or os.getenv("CI")
        or os.getenv("GITHUB_ACTIONS")
    )
    if replay_only or not (os.getenv("AWS_PROFILE") or os.getenv("AWS_BEARER_TOKEN_BEDROCK")):
        for name in ("AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN"):
            if replay_only or name not in os.environ:
                monkeypatch.setenv(name, "testing")
