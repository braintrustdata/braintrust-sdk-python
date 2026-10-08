"""Verify manual ADK wrapping covers Workflow nodes and direct tool calls."""

import asyncio

from braintrust import logger
from braintrust.integrations.adk import wrap_runner, wrap_workflow
from braintrust.integrations.adk.patchers import WorkflowNodeRunPatcher
from braintrust.test_helpers import init_test_logger
from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from google.adk.tools.load_artifacts_tool import LoadArtifactsTool
from google.adk.workflow import START, BaseNode, Workflow
from google.genai import types


async def main():
    app_name = "manual_workflow_app"
    user_id = "test-user"
    session_id = "manual-workflow-session"
    session_service = InMemorySessionService()
    await session_service.create_session(app_name=app_name, user_id=user_id, session_id=session_id)
    runner = Runner(
        agent=Workflow(name="manual_workflow", edges=[(START, LoadArtifactsTool())]),
        app_name=app_name,
        session_service=session_service,
    )
    async for _ in runner.run_async(
        user_id=user_id,
        session_id=session_id,
        new_message=types.Content(role="user", parts=[types.Part(text='{"artifact_names": ["report.txt"]}')]),
    ):
        pass


wrap_runner(Runner)
wrap_workflow(Workflow)
assert getattr(BaseNode, WorkflowNodeRunPatcher.patch_marker_attr(), False)

init_test_logger("manual-adk-workflow")
with logger._internal_with_memory_background_logger() as memory_logger:
    asyncio.run(main())
    spans = memory_logger.pop()

assert any(span["span_attributes"]["name"] == "invocation [manual_workflow_app]" for span in spans)
assert any(span["span_attributes"]["name"] == "workflow [manual_workflow]" for span in spans)
assert any(span["span_attributes"]["name"] == "tool [load_artifacts]" for span in spans)
print("SUCCESS")
