import dataclasses
import sys
from typing import Any, Literal, cast

import slugify

from .generated_types import IfExists
from .logger import BraintrustState, _state
from .util import response_raise_for_status


@dataclasses.dataclass
class SandboxConfig:
    """Configuration for a sandbox runtime."""

    provider: Literal["modal"]
    """The sandbox provider. Currently only "modal" is supported."""
    snapshot_ref: str
    """Reference to the sandbox snapshot."""


@dataclasses.dataclass
class RegisteredSandboxFunction:
    """Registered eval function discovered from sandbox list endpoint."""

    eval_name: str
    """Eval name discovered in the sandbox."""
    id: str
    """Unique identifier for the function."""
    name: str
    """Function name."""
    slug: str
    """URL-friendly identifier."""


@dataclasses.dataclass
class RegisterSandboxResult:
    """Result of registering a sandbox."""

    project_id: str
    """Project ID the sandbox is registered in."""
    functions: list[RegisteredSandboxFunction]
    """Registered eval functions discovered from this sandbox."""


SANDBOX_GROUP_NAME_METADATA_KEY = "_bt_sandbox_group_name"


def register_sandbox(
    name: str,
    project: str,
    sandbox: SandboxConfig,
    *,
    entrypoints: list[str] | None = None,
    description: str | None = None,
    metadata: dict[str, Any] | None = None,
    if_exists: IfExists | None = None,
    api_key: str | None = None,
    app_url: str | None = None,
    org_name: str | None = None,
    state: BraintrustState | None = None,
) -> RegisterSandboxResult:
    """Register a sandbox function with Braintrust.

    :param name: Group name for the sandbox functions.
    :param project: Name of the project to register the sandbox in.
    :param sandbox: Sandbox configuration (provider and snapshot reference).
    :param entrypoints: Optional list of entrypoints available in the sandbox.
    :param description: Optional description.
    :param metadata: Optional metadata dict.
    :param if_exists: What to do if function already exists. Defaults to "replace".
    :param api_key: Braintrust API key. Uses BRAINTRUST_API_KEY env var if not provided.
    :param app_url: Braintrust app URL. Uses default if not provided.
    :param org_name: Organization name.
    :param state: Optional BraintrustState instance. Defaults to the global state.
    :returns: RegisterSandboxResult with project_id and created eval functions.

    Example::

        from braintrust import register_sandbox, SandboxConfig

        result = register_sandbox(
            name="My Sandbox",
            project="My Project",
            entrypoints=["./my-eval.eval.py"],
            sandbox=SandboxConfig(provider="modal", snapshot_ref="sb-xxx"),
        )
        print([f.id for f in result.functions])
    """
    state = state or _state
    state.login(api_key=api_key, app_url=app_url, org_name=org_name)

    project_response = state.app_conn().post_json(
        "api/project/register", {"project_name": project, "org_id": state.org_id}
    )
    project_id = project_response["project"]["id"]

    runtime_context = {
        "runtime": "python",
        "version": f"{sys.version_info.major}.{sys.version_info.minor}",
    }

    list_response = state.proxy_conn().post(
        "function/sandbox-list",
        json={
            "sandbox_spec": {
                "provider": sandbox.provider,
                "snapshot_ref": sandbox.snapshot_ref,
            },
            "runtime_context": runtime_context,
            "entrypoints": entrypoints,
            "project_id": project_id,
        },
        headers={"x-bt-org-name": state.org_name},
    )
    response_raise_for_status(list_response)
    evaluator_definitions = cast(dict[str, Any], list_response.json())

    function_defs: list[dict[str, Any]] = []
    slug_to_eval_name: dict[str, str] = {}
    for eval_name, evaluator_definition in evaluator_definitions.items():
        slug = slugify.slugify(eval_name)
        slug_to_eval_name[slug] = eval_name
        function_def: dict[str, Any] = {
            "project_id": project_id,
            "org_name": state.org_name,
            "name": eval_name,
            "slug": slug,
            "function_type": "sandbox",
            "function_data": {
                "type": "code",
                "data": {
                    "type": "bundle",
                    "runtime_context": runtime_context,
                    "location": {
                        "type": "sandbox",
                        "sandbox_spec": {
                            "provider": sandbox.provider,
                            "snapshot_ref": sandbox.snapshot_ref,
                        },
                        "entrypoints": entrypoints,
                        "eval_name": eval_name,
                        "evaluator_definition": evaluator_definition,
                    },
                    "bundle_id": None,
                    "preview": None,
                },
            },
            "metadata": {
                **(metadata or {}),
                SANDBOX_GROUP_NAME_METADATA_KEY: name,
            },
            "if_exists": if_exists or "replace",
        }
        if description is not None:
            function_def["description"] = description
        function_defs.append(function_def)

    response = state.api_conn().post_json("insert-functions", {"functions": function_defs})

    functions: list[RegisteredSandboxFunction] = []
    for fn in response["functions"]:
        eval_name = slug_to_eval_name[fn["slug"]]
        functions.append(
            RegisteredSandboxFunction(
                eval_name=eval_name,
                id=fn["id"],
                name=eval_name,
                slug=fn["slug"],
            )
        )

    return RegisterSandboxResult(
        project_id=project_id,
        functions=functions,
    )
