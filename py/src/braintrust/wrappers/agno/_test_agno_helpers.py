# pyright: reportMissingParameterType=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownParameterType=false
# pyright: reportUnknownVariableType=false

from inspect import isawaitable

from braintrust.wrappers.agno.agent import wrap_agent

PROJECT_NAME = "test-agno-app"


class FakeMetrics:
    def __init__(self):
        self.input_tokens = 1
        self.output_tokens = 2
        self.total_tokens = 3
        self.duration = 0.1
        self.time_to_first_token = 0.01


class FakeRunOutput:
    def __init__(self, content: str):
        self.content = content
        self.status = "COMPLETED"
        self.model = "fake-model"
        self.model_provider = "FakeProvider"
        self.metrics = FakeMetrics()


class FakeEvent:
    def __init__(self, event: str, **kwargs):
        self.event = event
        for k, v in kwargs.items():
            setattr(self, k, v)


class FakeExecutionInput:
    def __init__(self, input):
        self.input = input
        self.kind = "workflow-execution"


class FakeWorkflowRunResponse:
    def __init__(self, input=None, content: str | None = None):
        self.input = input
        self.content = content
        self.status = "COMPLETED"
        self.metrics = FakeMetrics()


def make_fake_workflow(name: str):
    class FakeWorkflow:
        def __init__(self):
            self.name = name
            self.steps = ["first-step"]

        async def _aexecute(self, session_id, user_id, execution_input, workflow_run_response, run_context=None):
            return FakeWorkflowRunResponse(input=execution_input.input, content="workflow-async")

        def _execute_stream(self, session, execution_input, workflow_run_response, run_context=None):
            yield FakeEvent("WorkflowStarted", content=None)
            yield FakeEvent("StepStarted", content=None)
            yield FakeEvent("StepCompleted", content="hello ")
            yield FakeEvent("WorkflowCompleted", content="world", metrics=FakeMetrics(), status="COMPLETED")

    return FakeWorkflow


def make_fake_duplicate_content_workflow(name: str):
    class FakeWorkflow:
        def __init__(self):
            self.name = name
            self.steps = ["first-step"]

        def _execute_stream(self, session, execution_input, workflow_run_response, run_context=None):
            yield FakeEvent("StepCompleted", content="hello")
            yield FakeEvent("WorkflowCompleted", content="hello", metrics=FakeMetrics(), status="COMPLETED")

    return FakeWorkflow


def make_fake_streaming_workflow_with_mutated_run_response(name: str):
    class FakeWorkflow:
        def __init__(self):
            self.name = name
            self.steps = ["first-step"]

        def _execute_stream(self, session, execution_input, workflow_run_response, run_context=None):
            yield FakeEvent("WorkflowStarted", content=None)
            yield FakeEvent("StepCompleted", content="hello ")
            workflow_run_response.content = "world"
            workflow_run_response.status = "FAILED"
            workflow_run_response.metrics = FakeMetrics()
            yield FakeEvent("WorkflowCompleted", content="world")

    return FakeWorkflow


def make_fake_workflow_with_async_agent(name: str, agent_name: str):
    class FakeAgent:
        def __init__(self):
            self.name = agent_name

        async def arun(self, input, stream=False, **kwargs):
            return FakeRunOutput(f"{input}-async")

    WrappedAgent = wrap_agent(FakeAgent)

    class FakeWorkflow:
        def __init__(self):
            self.name = name
            self.id = "workflow-123"
            self.steps = ["agent-step"]
            self.agent = WrappedAgent()

        async def _aexecute(self, session_id, user_id, execution_input, workflow_run_response, run_context=None):
            return await self.agent.arun(execution_input.input)

    return FakeWorkflow


def make_fake_workflow_agent_path(name: str):
    class FakeWorkflow:
        def __init__(self):
            self.name = name
            self.id = "workflow-agent-123"
            self.steps = ["agent-step"]

        def _execute_workflow_agent(self, user_input, session, execution_input, run_context, stream=False, **kwargs):
            if stream:
                def _stream():
                    yield FakeEvent("WorkflowStarted")
                    yield FakeEvent(
                        "WorkflowCompleted",
                        content=f"{user_input}-sync-stream",
                        metrics=FakeMetrics(),
                        status="COMPLETED",
                    )

                return _stream()
            return FakeRunOutput(f"{user_input}-sync")

        async def _aexecute_workflow_agent(self, user_input, run_context, execution_input, stream=False, **kwargs):
            if stream:

                async def _astream():
                    yield FakeEvent("WorkflowStarted")
                    yield FakeEvent(
                        "WorkflowCompleted",
                        content=f"{user_input}-async-stream",
                        metrics=FakeMetrics(),
                        status="COMPLETED",
                    )

                return _astream()
            return FakeRunOutput(f"{user_input}-async")

    return FakeWorkflow


def make_fake_component(name: str):
    class FakeComponent:
        def __init__(self):
            self.name = name

        def run(self, input, stream=False, **kwargs):
            if stream:

                def _stream():
                    yield FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    yield FakeEvent("RunContent", content=f"{input}-sync")
                    yield FakeEvent("RunCompleted", metrics=FakeMetrics())

                return _stream()
            return FakeRunOutput(f"{input}-sync")

        def arun(self, input, stream=False, **kwargs):
            if stream:

                async def _astream():
                    yield FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    yield FakeEvent("RunContent", content=f"{input}-async")
                    yield FakeEvent("RunCompleted", metrics=FakeMetrics())

                return _astream()

            async def _result():
                return FakeRunOutput(f"{input}-async")

            return _result()

    return FakeComponent


def make_fake_async_dispatch_component(name: str):
    class FakeComponent:
        def __init__(self):
            self.name = name

        async def arun(self, input, stream=False, **kwargs):
            if stream:

                async def _astream():
                    yield FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    yield FakeEvent("RunContent", content=f"{input}-awaited-async")
                    yield FakeEvent("RunCompleted", metrics=FakeMetrics())

                return _astream()
            return {"content": f"{input}-awaited-async"}

    return FakeComponent


def make_fake_error_component(name: str):
    class FakeComponent:
        def __init__(self):
            self.name = name

        def run(self, input, stream=False, **kwargs):
            if stream:

                def _stream():
                    yield FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    raise RuntimeError("sync-stream-error")

                return _stream()
            return FakeRunOutput(f"{input}-sync")

        def arun(self, input, stream=False, **kwargs):
            if stream:

                async def _astream():
                    yield FakeEvent("RunStarted", model="fake-model", model_provider="FakeProvider")
                    raise RuntimeError("async-stream-error")

                return _astream()

            async def _result():
                return FakeRunOutput(f"{input}-async")

            return _result()

    return FakeComponent


def make_fake_private_public_component(name: str):
    class FakeComponent:
        def __init__(self):
            self.name = name
            self.calls = []

        def _run(self, run_response=None, run_messages=None, **kwargs):
            self.calls.append("_run")
            return FakeRunOutput("private-run")

        def run(self, input, **kwargs):
            self.calls.append("run")
            return FakeRunOutput("public-run")

        async def _arun(self, run_response=None, input=None, **kwargs):
            self.calls.append("_arun")
            return FakeRunOutput("private-arun")

        def arun(self, input, **kwargs):
            self.calls.append("arun")

            async def _result():
                return FakeRunOutput("public-arun")

            return _result()

    return FakeComponent


class StrictSpan:
    def __init__(self):
        self.ended = False

    def set_current(self):
        return None

    def unset_current(self):
        return None

    def log(self, **kwargs):
        if self.ended:
            raise AssertionError("log called after span.end()")

    def end(self):
        self.ended = True


__all__ = [
    "FakeExecutionInput",
    "FakeWorkflowRunResponse",
    "PROJECT_NAME",
    "StrictSpan",
    "isawaitable",
    "make_fake_async_dispatch_component",
    "make_fake_component",
    "make_fake_workflow_agent_path",
    "make_fake_workflow_with_async_agent",
    "make_fake_duplicate_content_workflow",
    "make_fake_error_component",
    "make_fake_private_public_component",
    "make_fake_workflow",
]
