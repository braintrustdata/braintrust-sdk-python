import argparse
import asyncio
import importlib
import inspect
import json
import pathlib
import statistics
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from .framework import call_user_fn
from .score import Score, ScoreLike, is_score
from .trace import GetThreadOptions, SpanData

__all__ = [
    "ReplayCase",
    "ReplayResult",
    "ReplaySummary",
    "ReplayTrace",
    "import_callable",
    "load_trace_file",
    "replay_traces",
]


class ReplayTask(Protocol):
    def __call__(self, input: Any, **kwargs: Any) -> Any | Awaitable[Any]: ...


class ReplayScorer(Protocol):
    def __call__(self, *args: Any, **kwargs: Any) -> Any | Awaitable[Any]: ...


@dataclass
class ReplayTrace:
    root_span_id: str
    spans: list[SpanData]
    configuration: dict[str, str] = field(default_factory=dict)

    def get_configuration(self) -> dict[str, str]:
        return self.configuration.copy()

    async def get_spans(
        self,
        span_type: list[str] | None = None,
        *,
        include_scorers: bool = False,
    ) -> list[SpanData]:
        spans = self.spans
        if not include_scorers:
            spans = [
                span
                for span in spans
                if not ((span.span_attributes or {}).get("purpose") == "scorer")
            ]
        if span_type:
            allowed_types = set(span_type)
            spans = [span for span in spans if (span.span_attributes or {}).get("type") in allowed_types]
        return list(spans)

    async def get_thread(self, options: GetThreadOptions | None = None) -> list[Any]:
        del options
        root = self.root_span
        if root is None:
            return []
        if isinstance(root.input, dict):
            for key in ("messages", "thread"):
                value = root.input.get(key)
                if isinstance(value, list):
                    return value
        return []

    @property
    def root_span(self) -> SpanData | None:
        for span in self.spans:
            if getattr(span, "is_root", False) or span.span_id == self.root_span_id:
                return span
        return self.spans[0] if self.spans else None


@dataclass
class ReplayCase:
    trace: ReplayTrace
    input: Any
    expected: Any | None = None
    baseline_output: Any | None = None
    metadata: dict[str, Any] | None = None
    tags: Sequence[str] | None = None


@dataclass
class ReplayResult:
    root_span_id: str
    input: Any
    output: Any
    baseline_output: Any | None
    expected: Any | None
    scores: dict[str, float | None]
    baseline_scores: dict[str, float | None]
    score_deltas: dict[str, float | None]
    metrics: dict[str, Any]
    baseline_metrics: dict[str, Any]
    metric_deltas: dict[str, float]
    span_count: int
    error: str | None = None


@dataclass
class ReplaySummary:
    results: list[ReplayResult]

    @property
    def failed_count(self) -> int:
        return sum(1 for result in self.results if result.error is not None)

    @property
    def score_averages(self) -> dict[str, float | None]:
        score_names = sorted({name for result in self.results for name in result.scores})
        averages: dict[str, float | None] = {}
        for name in score_names:
            values = [result.scores[name] for result in self.results if result.scores.get(name) is not None]
            averages[name] = statistics.fmean(values) if values else None
        return averages

    @property
    def score_delta_averages(self) -> dict[str, float | None]:
        score_names = sorted({name for result in self.results for name in result.score_deltas})
        averages: dict[str, float | None] = {}
        for name in score_names:
            values = [
                result.score_deltas[name]
                for result in self.results
                if result.score_deltas.get(name) is not None
            ]
            averages[name] = statistics.fmean(values) if values else None
        return averages

    def as_dict(self) -> dict[str, Any]:
        return {
            "summary": {
                "trace_count": len(self.results),
                "failed_count": self.failed_count,
                "score_averages": self.score_averages,
                "score_delta_averages": self.score_delta_averages,
            },
            "results": [
                {
                    "root_span_id": result.root_span_id,
                    "input": result.input,
                    "output": result.output,
                    "baseline_output": result.baseline_output,
                    "expected": result.expected,
                    "scores": result.scores,
                    "baseline_scores": result.baseline_scores,
                    "score_deltas": result.score_deltas,
                    "metrics": result.metrics,
                    "baseline_metrics": result.baseline_metrics,
                    "metric_deltas": result.metric_deltas,
                    "span_count": result.span_count,
                    "error": result.error,
                }
                for result in self.results
            ],
        }

    def check_thresholds(
        self,
        *,
        min_scores: Mapping[str, float] | None = None,
        min_score_deltas: Mapping[str, float] | None = None,
        fail_on_error: bool = False,
    ) -> list[str]:
        failures: list[str] = []
        if fail_on_error and self.failed_count:
            failures.append(f"{self.failed_count} trace replay(s) failed")
        for name, minimum in (min_scores or {}).items():
            actual = self.score_averages.get(name)
            if actual is None:
                failures.append(f"score {name!r} was not produced")
            elif actual < minimum:
                failures.append(f"score {name!r} averaged {actual:.4f}, below required {minimum:.4f}")
        for name, minimum in (min_score_deltas or {}).items():
            actual = self.score_delta_averages.get(name)
            if actual is None:
                failures.append(f"score delta {name!r} was not produced")
            elif actual < minimum:
                failures.append(f"score delta {name!r} averaged {actual:+.4f}, below required {minimum:+.4f}")
        return failures


def load_trace_file(path: str | pathlib.Path) -> list[ReplayCase]:
    path = pathlib.Path(path)
    spans = [_span_from_dict(row) for row in _load_span_rows(path)]
    grouped: dict[str, list[SpanData]] = {}
    for span in spans:
        root_span_id = getattr(span, "root_span_id", None) or span.span_id
        if root_span_id is None:
            raise ValueError(f"Span row in {path} is missing both root_span_id and span_id")
        grouped.setdefault(root_span_id, []).append(span)

    cases: list[ReplayCase] = []
    for root_span_id, trace_spans in grouped.items():
        trace = ReplayTrace(
            root_span_id=root_span_id,
            spans=trace_spans,
            configuration={"object_type": "local_trace", "object_id": str(path), "root_span_id": root_span_id},
        )
        root = trace.root_span
        if root is None:
            continue
        cases.append(
            ReplayCase(
                trace=trace,
                input=root.input,
                expected=root.expected,
                baseline_output=root.output,
                metadata=root.metadata,
                tags=root.tags,
            )
        )
    return cases


async def replay_traces(
    cases: Sequence[ReplayCase],
    *,
    task: ReplayTask | None = None,
    scorers: Sequence[ReplayScorer] | None = None,
) -> ReplaySummary:
    event_loop = asyncio.get_event_loop()
    results = [
        await _replay_case(event_loop, case, task=task, scorers=list(scorers or []))
        for case in cases
    ]
    return ReplaySummary(results=results)


def import_callable(spec: str) -> Callable[..., Any]:
    if ":" not in spec:
        raise ValueError(f"Expected callable path in module:function form, got {spec!r}")
    module_name, attr_path = spec.split(":", 1)
    obj: Any = importlib.import_module(module_name)
    for attr in attr_path.split("."):
        obj = getattr(obj, attr)
    if inspect.isclass(obj):
        obj = obj()
    if not callable(obj):
        raise TypeError(f"{spec!r} does not resolve to a callable")
    return obj


def build_parser(subparsers: argparse._SubParsersAction, parent_parser: argparse.ArgumentParser) -> None:
    parser = subparsers.add_parser(
        "replay",
        help="Replay local trace exports as eval regression cases.",
        parents=[parent_parser],
    )
    parser.add_argument("trace_file", help="Path to a JSON or JSONL trace export.")
    parser.add_argument("--task", help="Callable to rerun for each root-span input, in module:function form.")
    parser.add_argument(
        "--score",
        action="append",
        default=[],
        help="Scorer callable in module:function form. May be specified more than once.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit the replay report as JSON.",
    )
    parser.add_argument(
        "--min-score",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Require an average score to meet a minimum value. May be specified more than once.",
    )
    parser.add_argument(
        "--min-score-delta",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Require an average score delta versus the trace baseline. May be specified more than once.",
    )
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any trace replay fails.",
    )
    parser.set_defaults(func=run_cli)


def run_cli(args: argparse.Namespace) -> None:
    task = import_callable(args.task) if args.task else None
    scorers = [import_callable(spec) for spec in args.score]
    summary = asyncio.run(replay_traces(load_trace_file(args.trace_file), task=task, scorers=scorers))
    threshold_failures = summary.check_thresholds(
        min_scores=_parse_thresholds(args.min_score),
        min_score_deltas=_parse_thresholds(args.min_score_delta),
        fail_on_error=args.fail_on_error,
    )
    if args.json:
        report = summary.as_dict()
        report["checks"] = {"passed": not threshold_failures, "failures": threshold_failures}
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        _print_human_report(summary, threshold_failures)
    if threshold_failures:
        raise SystemExit(1)


async def _replay_case(
    event_loop: asyncio.AbstractEventLoop,
    case: ReplayCase,
    *,
    task: ReplayTask | None,
    scorers: list[ReplayScorer],
) -> ReplayResult:
    output = case.baseline_output
    error: str | None = None
    try:
        if task is not None:
            output = await _call_task(event_loop, task, case)
        scores = await _score_case(event_loop, case, output, scorers)
    except Exception as err:
        scores = {}
        error = f"{type(err).__name__}: {err}"

    baseline_scores = _numeric_mapping(case.trace.root_span.scores if case.trace.root_span else None)
    baseline_metrics = _numeric_mapping(case.trace.root_span.metrics if case.trace.root_span else None)
    metrics = _derive_metrics(case.trace.spans)
    return ReplayResult(
        root_span_id=case.trace.root_span_id,
        input=case.input,
        output=output,
        baseline_output=case.baseline_output,
        expected=case.expected,
        scores=scores,
        baseline_scores=baseline_scores,
        score_deltas=_delta(scores, baseline_scores),
        metrics=metrics,
        baseline_metrics=baseline_metrics,
        metric_deltas=_delta(metrics, baseline_metrics, include_only_common=True),
        span_count=len(case.trace.spans),
        error=error,
    )


async def _call_task(
    event_loop: asyncio.AbstractEventLoop,
    task: ReplayTask,
    case: ReplayCase,
) -> Any:
    kwargs = {
        "input": case.input,
        "expected": case.expected,
        "metadata": dict(case.metadata or {}),
        "trace": case.trace,
    }
    return await call_user_fn(event_loop, task, **kwargs)


async def _score_case(
    event_loop: asyncio.AbstractEventLoop,
    case: ReplayCase,
    output: Any,
    scorers: Sequence[ReplayScorer],
) -> dict[str, float | None]:
    scores: dict[str, float | None] = {}
    for index, scorer in enumerate(scorers):
        name = _callable_name(scorer, index)
        kwargs = {
            "input": case.input,
            "output": output,
            "expected": case.expected,
            "metadata": dict(case.metadata or {}),
            "trace": case.trace,
        }
        result = await call_user_fn(event_loop, scorer, **kwargs)
        for score in _coerce_scores(name, result):
            scores[score.name] = score.score
    return scores


def _coerce_scores(name: str, result: Any) -> list[ScoreLike]:
    if result is None:
        return [Score(name=name, score=result)]
    if isinstance(result, bool):
        return [Score(name=name, score=float(result))]
    if isinstance(result, (float, int)):
        return [Score(name=name, score=result)]
    if isinstance(result, dict):
        return [Score.from_dict(result)]
    if is_score(result):
        return [result]
    if isinstance(result, Iterable) and not isinstance(result, (str, bytes, Mapping)):
        coerced = []
        for item in result:
            if isinstance(item, dict):
                item = Score.from_dict(item)
            if not is_score(item):
                raise ValueError(f"Scorer {name!r} returned an invalid score item: {item!r}")
            coerced.append(item)
        return coerced
    raise ValueError(f"Scorer {name!r} returned an invalid score: {result!r}")


def _load_span_rows(path: pathlib.Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    raw = json.loads(text)
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict):
        for key in ("spans", "rows"):
            value = raw.get(key)
            if isinstance(value, list):
                return value
        trace = raw.get("trace")
        if isinstance(trace, dict) and isinstance(trace.get("spans"), list):
            return trace["spans"]
    raise ValueError(f"{path} must contain a list of spans, a JSONL stream, or an object with a spans field")


def _parse_thresholds(values: Sequence[str]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(f"Expected threshold in NAME=VALUE form, got {value!r}")
        name, raw_threshold = value.split("=", 1)
        if not name:
            raise ValueError(f"Threshold name cannot be empty in {value!r}")
        thresholds[name] = float(raw_threshold)
    return thresholds


def _span_from_dict(row: dict[str, Any]) -> SpanData:
    return SpanData.from_dict(row)


def _numeric_mapping(value: Any) -> dict[str, float | None]:
    if not isinstance(value, Mapping):
        return {}
    result: dict[str, float | None] = {}
    for key, item in value.items():
        if item is None or (isinstance(item, (float, int)) and not isinstance(item, bool)):
            result[str(key)] = item
    return result


def _derive_metrics(spans: Sequence[SpanData]) -> dict[str, Any]:
    metrics: dict[str, Any] = {"span_count": len(spans)}
    totals: dict[str, float] = {}
    for span in spans:
        for key, value in _numeric_mapping(span.metrics).items():
            if value is not None:
                totals[key] = totals.get(key, 0.0) + value
    metrics.update(totals)
    return metrics


def _delta(
    current: Mapping[str, float | None],
    baseline: Mapping[str, float | None],
    *,
    include_only_common: bool = False,
) -> dict[str, float | None]:
    keys = set(current) & set(baseline) if include_only_common else set(current) | set(baseline)
    result: dict[str, float | None] = {}
    for key in sorted(keys):
        current_value = current.get(key)
        baseline_value = baseline.get(key)
        if current_value is None or baseline_value is None:
            result[key] = None
        else:
            result[key] = current_value - baseline_value
    return result


def _callable_name(fn: Callable[..., Any], index: int) -> str:
    return getattr(fn, "__name__", None) or fn.__class__.__name__ or f"score_{index}"


def _print_human_report(summary: ReplaySummary, threshold_failures: Sequence[str] = ()) -> None:
    print(f"Replayed {len(summary.results)} trace(s), {summary.failed_count} failed")
    if summary.score_averages:
        print("Score averages:")
        for name, value in summary.score_averages.items():
            formatted = "skipped" if value is None else f"{value:.4f}"
            print(f"  {name}: {formatted}")
    if summary.score_delta_averages:
        print("Score delta averages:")
        for name, value in summary.score_delta_averages.items():
            formatted = "n/a" if value is None else f"{value:+.4f}"
            print(f"  {name}: {formatted}")
    for result in summary.results:
        if result.error:
            print(f"- {result.root_span_id}: {result.error}")
    if threshold_failures:
        print("Replay checks failed:")
        for failure in threshold_failures:
            print(f"  {failure}")
