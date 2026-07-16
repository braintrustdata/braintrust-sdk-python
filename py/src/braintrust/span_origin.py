import importlib.metadata
import os
from typing import Any, TypedDict

from .env import BraintrustEnv


class SpanOriginEnvironment(TypedDict, total=False):
    type: str
    name: str


def detect_environment(
    explicit: SpanOriginEnvironment | None = None,
) -> SpanOriginEnvironment | None:
    if explicit:
        return explicit

    env_type = BraintrustEnv.ENVIRONMENT_TYPE.get(None, use_dotenv=True)
    env_name = BraintrustEnv.ENVIRONMENT_NAME.get(None, use_dotenv=True)
    if env_type or env_name:
        return {
            **({"type": env_type} if env_type else {}),
            **({"name": env_name} if env_name else {}),
        }

    ci = _first_present(
        {
            "GITHUB_ACTIONS": "github_actions",
            "GITLAB_CI": "gitlab_ci",
            "CIRCLECI": "circleci",
            "BUILDKITE": "buildkite",
            "JENKINS_URL": "jenkins",
            "JENKINS_HOME": "jenkins",
            "TF_BUILD": "azure_pipelines",
            "TEAMCITY_VERSION": "teamcity",
            "TRAVIS": "travis",
            "BITBUCKET_BUILD_NUMBER": "bitbucket",
        }
    )
    if ci:
        return {"type": "ci", "name": ci}
    if os.environ.get("CI"):
        return {"type": "ci", "name": "ci"}

    server = _first_present(
        {
            "VERCEL": "vercel",
            "NETLIFY": "netlify",
        }
    )
    if server:
        return {"type": "server", "name": server}

    if os.environ.get("ECS_CONTAINER_METADATA_URI") or os.environ.get("ECS_CONTAINER_METADATA_URI_V4"):
        return {"type": "server", "name": "ecs"}
    aws_execution_env = os.environ.get("AWS_EXECUTION_ENV")
    if aws_execution_env:
        if aws_execution_env.startswith("AWS_ECS_"):
            return {"type": "server", "name": "ecs"}
        if aws_execution_env.startswith("AWS_Lambda_"):
            return {"type": "server", "name": "aws_lambda"}
    if os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        return {"type": "server", "name": "aws_lambda"}

    server = _first_present(
        {
            "K_SERVICE": "cloud_run",
            "FUNCTION_TARGET": "gcp_functions",
            "KUBERNETES_SERVICE_HOST": "kubernetes",
            "DYNO": "heroku",
            "FLY_APP_NAME": "fly",
            "RAILWAY_ENVIRONMENT": "railway",
            "RENDER_SERVICE_NAME": "render",
        }
    )
    if server:
        return {"type": "server", "name": server}

    return _deployment_mode(os.environ.get("PYTHON_ENV"))


def merge_span_origin_context(
    context: dict[str, Any] | None,
    instrumentation_name: str,
    environment: SpanOriginEnvironment | None,
) -> dict[str, Any]:
    merged = dict(context or {})
    span_origin = dict(merged.get("span_origin") or {})
    span_origin = {
        "name": "braintrust.sdk.python",
        "version": _sdk_version(),
        "instrumentation": {"name": instrumentation_name},
        **({"environment": environment} if environment else {}),
        **span_origin,
    }
    merged["span_origin"] = span_origin
    return merged


def _sdk_version() -> str:
    try:
        return importlib.metadata.version("braintrust")
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def _first_present(mapping: dict[str, str]) -> str | None:
    for key, name in mapping.items():
        if os.environ.get(key):
            return name
    return None


def _deployment_mode(value: str | None) -> SpanOriginEnvironment | None:
    if not value:
        return None
    normalized = value.lower()
    if normalized in ("production", "staging"):
        return {"type": "server", "name": normalized}
    if normalized in ("development", "local"):
        return {"type": "local", "name": normalized}
    return {"name": value}
