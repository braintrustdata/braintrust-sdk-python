import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

import pytest
from braintrust import logger
from braintrust.integrations.agentscope import setup_agentscope, wrap_evaluator
from braintrust.integrations.agentscope.patchers import (
    AgentCallPatcher,
    MetricCallPatcher,
    TaskEvaluatePatcher,
    _GeneralEvaluatorRunEvaluationPatcher,
    _GeneralEvaluatorRunPatcher,
    _GeneralEvaluatorRunSolutionPatcher,
)
from braintrust.span_types import SpanTypeAttribute
from braintrust.test_helpers import init_test_logger
from braintrust.wrappers.test_utils import verify_autoinstrument_script


PROJECT_NAME = "test_agentscope"

setup_agentscope(project_name=PROJECT_NAME)


@pytest.fixture(scope="module")
def vcr_config():
    return {
        "cassette_library_dir": str(Path(__file__).parent / "cassettes"),
    }


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


def _span_type(span):
    span_type = span["span_attributes"]["type"]
    return span_type.value if hasattr(span_type, "value") else span_type


def _make_model(*, stream: bool = False):
    from agentscope.model import OpenAIChatModel

    return OpenAIChatModel(
        model_name="gpt-4o-mini",
        stream=stream,
        generate_kwargs={"temperature": 0},
    )


def _make_agent(name: str, sys_prompt: str, *, toolkit=None, multi_agent: bool = False):
    from agentscope.agent import ReActAgent
    from agentscope.formatter import OpenAIChatFormatter, OpenAIMultiAgentFormatter
    from agentscope.memory import InMemoryMemory
    from agentscope.tool import Toolkit

    agent = ReActAgent(
        name=name,
        sys_prompt=sys_prompt,
        model=_make_model(),
        formatter=OpenAIMultiAgentFormatter() if multi_agent else OpenAIChatFormatter(),
        toolkit=toolkit or Toolkit(),
        memory=InMemoryMemory(),
    )
    if hasattr(agent, "set_console_output_enabled"):
        agent.set_console_output_enabled(False)
    elif hasattr(agent, "disable_console_output"):
        agent.disable_console_output()
    return agent


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_simple_agent_run(memory_logger):
    from agentscope.message import Msg

    assert not memory_logger.pop()

    agent = _make_agent(
        "Friday",
        "You are a concise assistant. Answer in one sentence.",
    )

    response = await agent(
        Msg(
            name="user",
            content="Say hello in exactly two words.",
            role="user",
        )
    )

    assert response is not None

    spans = memory_logger.pop()
    agent_span = next(span for span in spans if span["span_attributes"]["name"] == "Friday.reply")
    llm_spans = [span for span in spans if _span_type(span) == SpanTypeAttribute.LLM]

    assert _span_type(agent_span) == "task"
    assert llm_spans
    assert llm_spans[0]["metadata"]["model"] == "gpt-4o-mini"
    assert "args" not in llm_spans[0]["input"]
    assert llm_spans[0]["input"]["messages"][0]["role"] == "system"
    assert llm_spans[0]["input"]["messages"][1]["role"] == "user"
    assert llm_spans[0]["input"]["messages"][1]["content"][0]["text"] == "Say hello in exactly two words."
    assert llm_spans[0]["output"]["role"] == "assistant"
    assert llm_spans[0]["output"]["content"][0]["text"] == "Hello there."
    assert "usage" not in llm_spans[0]["output"]
    assert agent_span["span_id"] in llm_spans[0]["span_parents"]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_sequential_pipeline_creates_parent_span(memory_logger):
    from agentscope.message import Msg
    from agentscope.pipeline import sequential_pipeline

    assert not memory_logger.pop()

    agents = [
        _make_agent("Alice", "You rewrite the input as a short title.", multi_agent=True),
        _make_agent("Bob", "You answer the previous message in one sentence.", multi_agent=True),
    ]

    result = await sequential_pipeline(
        agents=agents,
        msg=Msg(
            name="user",
            content="Summarize why tests should use real recorded traffic.",
            role="user",
        ),
    )

    assert result is not None

    spans = memory_logger.pop()
    pipeline_span = next(span for span in spans if span["span_attributes"]["name"] == "sequential_pipeline.run")
    alice_span = next(span for span in spans if span["span_attributes"]["name"] == "Alice.reply")
    bob_span = next(span for span in spans if span["span_attributes"]["name"] == "Bob.reply")

    assert _span_type(pipeline_span) == "task"
    assert pipeline_span["span_id"] in alice_span["span_parents"]
    assert pipeline_span["span_id"] in bob_span["span_parents"]


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_tool_use_creates_tool_span(memory_logger):
    from agentscope.message import Msg
    from agentscope.tool import Toolkit, execute_python_code

    assert not memory_logger.pop()

    toolkit = Toolkit()
    toolkit.register_tool_function(execute_python_code)
    agent = _make_agent(
        "Jarvis",
        "You are a helpful assistant. Use tools when required and keep answers brief.",
        toolkit=toolkit,
    )

    response = await agent(
        Msg(
            name="user",
            content="Use Python to compute 6 * 7 and return just the result.",
            role="user",
        )
    )

    assert response is not None

    spans = memory_logger.pop()
    tool_spans = [span for span in spans if _span_type(span) == "tool"]

    assert tool_spans
    assert tool_spans[0]["span_attributes"]["name"] == "execute_python_code.execute"
    assert tool_spans[0]["input"]["tool_name"] == "execute_python_code"
    assert tool_spans[0]["output"]["content"]

    llm_spans = [span for span in spans if _span_type(span) == SpanTypeAttribute.LLM]
    assert llm_spans
    assert llm_spans[0]["output"]["role"] == "assistant"
    assert llm_spans[0]["output"]["content"][0]["type"] == "tool_use"
    assert "usage" not in llm_spans[0]["output"]


@pytest.mark.asyncio
async def test_model_call_wrapper_stream_logs_final_output_and_metrics(memory_logger):
    from braintrust.integrations.agentscope.tracing import _model_call_wrapper

    assert not memory_logger.pop()

    class FakeOpenAIChatModel:
        model_name = "gpt-4o-mini"

    async def wrapped(*_args, **_kwargs):
        async def _stream():
            yield {"content": [{"type": "text", "text": "Hello"}]}
            yield {
                "content": [{"type": "text", "text": "Hello there!"}],
                "usage": {"prompt_tokens": 29, "completion_tokens": 3, "total_tokens": 32},
            }

        return _stream()

    stream = await _model_call_wrapper(
        wrapped,
        FakeOpenAIChatModel(),
        args=([{"role": "user", "content": "Say hi in two words."}],),
        kwargs={},
    )

    chunks = [chunk async for chunk in stream]

    assert chunks[-1]["content"][0]["text"] == "Hello there!"

    spans = memory_logger.pop()
    assert len(spans) == 1
    llm_span = spans[0]

    assert _span_type(llm_span) == SpanTypeAttribute.LLM
    assert llm_span["output"]["role"] == "assistant"
    assert llm_span["output"]["content"][0]["text"] == "Hello there!"
    assert llm_span["metrics"]["prompt_tokens"] == 29
    assert llm_span["metrics"]["completion_tokens"] == 3
    assert llm_span["metrics"]["tokens"] == 32


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_agentscope_general_evaluator_creates_eval_spans(memory_logger, tmp_path):
    from agentscope.evaluate import (
        BenchmarkBase,
        FileEvaluatorStorage,
        GeneralEvaluator,
        MetricBase,
        MetricResult,
        MetricType,
        SolutionOutput,
        Task,
    )
    from agentscope.message import Msg

    assert not memory_logger.pop()

    class ExactMatchMetric(MetricBase):
        def __init__(self, ground_truth: str):
            super().__init__(
                name="exact_match",
                metric_type=MetricType.NUMERICAL,
                description="Check whether the model answer exactly matches the ground truth.",
                categories=[],
            )
            self.ground_truth = ground_truth

        async def __call__(self, solution: SolutionOutput) -> MetricResult:
            is_match = solution.output == self.ground_truth
            return MetricResult(
                name=self.name,
                result=1.0 if is_match else 0.0,
                message="Correct" if is_match else "Incorrect",
            )

    class ToyBenchmark(BenchmarkBase):
        def __init__(self, tasks):
            super().__init__(
                name="Toy benchmark",
                description="A one-task benchmark for AgentScope eval instrumentation.",
            )
            self.tasks = tasks

        def __iter__(self):
            yield from self.tasks

        def __len__(self):
            return len(self.tasks)

        def __getitem__(self, index):
            return self.tasks[index]

    task = Task(
        id="hello-task",
        input="Say hello in exactly two words.",
        ground_truth="Hello there.",
        metrics=[ExactMatchMetric("Hello there.")],
        tags={"difficulty": "easy", "category": "greeting"},
        metadata={"suite": "toy"},
    )
    evaluator = GeneralEvaluator(
        name="Toy benchmark evaluation",
        benchmark=ToyBenchmark([task]),
        n_repeat=1,
        storage=FileEvaluatorStorage(save_dir=str(tmp_path / "agentscope-eval")),
        n_workers=1,
    )

    async def solution(eval_task: Task, pre_hook):
        agent = _make_agent(
            "Friday",
            "You are a concise assistant. Answer in one sentence.",
        )
        if hasattr(agent, "register_instance_hook"):
            agent.register_instance_hook("pre_print", "save_logging", pre_hook)

        response = await agent(
            Msg(
                name="user",
                content=eval_task.input,
                role="user",
            )
        )

        content = response.content
        if isinstance(content, list):
            output = next(
                (item["text"] for item in content if isinstance(item, dict) and item.get("type") == "text"),
                None,
            )
            trajectory = content
        else:
            output = content
            trajectory = [content]

        return SolutionOutput(
            success=True,
            output=output,
            trajectory=trajectory,
            meta={"agent": "Friday"},
        )

    await evaluator.run(solution)

    spans = memory_logger.pop()
    root_span = next(span for span in spans if span["span_attributes"]["name"] == "agentscope.evaluate.run")
    solution_span = next(span for span in spans if span["span_attributes"]["name"] == "hello-task.solution")
    evaluation_span = next(span for span in spans if span["span_attributes"]["name"] == "hello-task.evaluate")
    metric_span = next(span for span in spans if span["span_attributes"]["name"] == "exact_match")
    agent_span = next(span for span in spans if span["span_attributes"]["name"] == "Friday.reply")

    assert _span_type(root_span) == "eval"
    assert root_span["metadata"]["benchmark_name"] == "Toy benchmark"
    assert root_span["metadata"]["task_count"] == 1
    assert root_span["output"]["status"] == "completed"

    assert _span_type(solution_span) == "task"
    assert solution_span["input"] == "Say hello in exactly two words."
    assert solution_span["expected"] == "Hello there."
    assert solution_span["tags"] == ["category:greeting", "difficulty:easy"]
    assert solution_span["metadata"]["repeat_id"] == "0"
    assert solution_span["metadata"]["metric_names"] == ["exact_match"]
    assert solution_span["metadata"]["task_tags"] == {"difficulty": "easy", "category": "greeting"}
    assert solution_span["output"]["output"] == "Hello there."
    assert solution_span["span_id"] in agent_span["span_parents"]

    assert _span_type(evaluation_span) == "eval"
    assert evaluation_span["span_id"] in metric_span["span_parents"]
    assert solution_span["span_id"] in evaluation_span["span_parents"]
    assert root_span["span_id"] in solution_span["span_parents"]
    assert evaluation_span["output"][0]["result"] == 1.0
    assert evaluation_span["output"][0]["message"] == "Correct"

    assert _span_type(metric_span) == "score"
    assert metric_span["scores"]["exact_match"] == 1.0
    assert metric_span["output"]["result"] == 1.0
    assert metric_span["output"]["message"] == "Correct"


@dataclass
class _FakeAgentscopeModules:
    AgentBase: type
    GeneralEvaluator: type
    MetricBase: type
    Task: type


@pytest.fixture
def fake_agentscope_modules(monkeypatch):
    agentscope_module = ModuleType("agentscope")
    agentscope_module.__path__ = []
    agentscope_module.__version__ = "1.0.0"

    agent_module = ModuleType("agentscope.agent")
    evaluate_module = ModuleType("agentscope.evaluate")

    class AgentBase:
        async def __call__(self, *_args, **_kwargs):
            return "ok"

    class Task:
        async def evaluate(self, *_args, **_kwargs):
            return []

    class MetricBase:
        async def __call__(self, *_args, **_kwargs):
            return None

    class GeneralEvaluator:
        async def run(self, *_args, **_kwargs):
            return None

        async def run_solution(self, *_args, **_kwargs):
            return None

        async def run_evaluation(self, *_args, **_kwargs):
            return None

    agent_module.AgentBase = AgentBase
    evaluate_module.GeneralEvaluator = GeneralEvaluator
    evaluate_module.Task = Task
    evaluate_module.MetricBase = MetricBase

    agentscope_module.agent = agent_module
    agentscope_module.evaluate = evaluate_module

    monkeypatch.setitem(sys.modules, "agentscope", agentscope_module)
    monkeypatch.setitem(sys.modules, "agentscope.agent", agent_module)
    monkeypatch.setitem(sys.modules, "agentscope.evaluate", evaluate_module)

    return _FakeAgentscopeModules(
        AgentBase=AgentBase,
        GeneralEvaluator=GeneralEvaluator,
        MetricBase=MetricBase,
        Task=Task,
    )


def test_setup_agentscope_can_skip_eval_patchers(fake_agentscope_modules):
    result = setup_agentscope(project_name=PROJECT_NAME, instrument_evals=False)

    assert result is True
    assert getattr(fake_agentscope_modules.AgentBase.__call__, AgentCallPatcher.patch_marker_attr(), False)
    assert not getattr(
        fake_agentscope_modules.GeneralEvaluator, _GeneralEvaluatorRunPatcher.patch_marker_attr(), False
    )
    assert not getattr(
        fake_agentscope_modules.GeneralEvaluator,
        _GeneralEvaluatorRunSolutionPatcher.patch_marker_attr(),
        False,
    )
    assert not getattr(
        fake_agentscope_modules.GeneralEvaluator,
        _GeneralEvaluatorRunEvaluationPatcher.patch_marker_attr(),
        False,
    )
    assert not getattr(fake_agentscope_modules.Task, TaskEvaluatePatcher.patch_marker_attr(), False)
    assert not getattr(fake_agentscope_modules.MetricBase, MetricCallPatcher.patch_marker_attr(), False)


def test_wrap_evaluator_patches_evaluator_and_eval_types(fake_agentscope_modules):
    wrapped = wrap_evaluator(fake_agentscope_modules.GeneralEvaluator)
    wrapped_again = wrap_evaluator(fake_agentscope_modules.GeneralEvaluator)

    assert wrapped is fake_agentscope_modules.GeneralEvaluator
    assert wrapped_again is fake_agentscope_modules.GeneralEvaluator
    assert getattr(fake_agentscope_modules.GeneralEvaluator, _GeneralEvaluatorRunPatcher.patch_marker_attr(), False)
    assert getattr(
        fake_agentscope_modules.GeneralEvaluator, _GeneralEvaluatorRunSolutionPatcher.patch_marker_attr(), False
    )
    assert getattr(
        fake_agentscope_modules.GeneralEvaluator,
        _GeneralEvaluatorRunEvaluationPatcher.patch_marker_attr(),
        False,
    )
    assert getattr(fake_agentscope_modules.Task, TaskEvaluatePatcher.patch_marker_attr(), False)
    assert getattr(fake_agentscope_modules.MetricBase, MetricCallPatcher.patch_marker_attr(), False)


class TestAutoInstrumentAgentScope:
    def test_auto_instrument_agentscope(self):
        verify_autoinstrument_script("test_auto_agentscope.py")
