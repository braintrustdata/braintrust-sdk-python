"""Type-check tests for asynchronous function invocation."""

from braintrust import BraintrustStream, invoke_async


async def _check_invoke_async_return_types() -> None:
    output: dict[str, str] = await invoke_async(project_name="project", slug="function", input={})
    stream: BraintrustStream = await invoke_async(project_name="project", slug="function", input={}, stream=True)

    assert output is not None
    async for chunk in stream:
        assert chunk is not None
