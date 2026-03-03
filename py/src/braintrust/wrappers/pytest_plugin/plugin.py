"""Braintrust pytest plugin.

Automatically tracks test results as Braintrust experiments when tests are
decorated with ``@pytest.mark.braintrust`` and run with ``--braintrust``.

The plugin is registered via the ``pytest11`` entry point in ``setup.py``
so it auto-loads when braintrust is installed.
"""

from __future__ import annotations

import traceback
from typing import Any

import pytest
from braintrust.logger import NOOP_SPAN, Span

# ---------------------------------------------------------------------------
# User-facing span wrapper
# ---------------------------------------------------------------------------


class BraintrustTestSpan:
    """Pytest-friendly wrapper around a :class:`braintrust.logger.Span`.

    Exposed to tests via the ``braintrust_span`` fixture.
    """

    _span: Span

    def __init__(self, span: Span) -> None:
        self._span = span

    def log_input(self, input: Any) -> None:
        """Log the test input."""
        self._span.log(input=input)

    def log_output(self, output: Any) -> None:
        """Log the test output."""
        self._span.log(output=output)

    def log_expected(self, expected: Any) -> None:
        """Log the expected output for the test."""
        self._span.log(expected=expected)

    def log_score(self, name: str, score: float) -> None:
        """Log a named score for the test."""
        self._span.log(scores={name: score})

    def log_metadata(self, metadata: dict[str, Any]) -> None:
        """Log metadata for the test."""
        self._span.log(metadata=metadata)

    @property
    def span(self) -> Span:
        """Escape hatch — the underlying :class:`~braintrust.logger.Span`."""
        return self._span


class _NoopBraintrustTestSpan(BraintrustTestSpan):
    """No-op variant returned when ``--braintrust`` is not active."""

    def __init__(self) -> None:
        super().__init__(NOOP_SPAN)


_NOOP_TEST_SPAN = _NoopBraintrustTestSpan()

# ---------------------------------------------------------------------------
# Marker registration & CLI options (always active)
# ---------------------------------------------------------------------------


def pytest_addoption(parser: pytest.Parser) -> None:
    group = parser.getgroup("braintrust", "Braintrust experiment tracking")
    group.addoption(
        "--braintrust",
        action="store_true",
        default=False,
        help="Enable Braintrust experiment tracking for @pytest.mark.braintrust tests.",
    )
    group.addoption(
        "--braintrust-project",
        action="store",
        default=None,
        help="Override the Braintrust project name for all tests (env: BRAINTRUST_PROJECT).",
    )
    group.addoption(
        "--braintrust-experiment",
        action="store",
        default=None,
        help="Override the experiment name (env: BRAINTRUST_EXPERIMENT).",
    )
    group.addoption(
        "--braintrust-api-key",
        action="store",
        default=None,
        help="Braintrust API key (env: BRAINTRUST_API_KEY).",
    )
    group.addoption(
        "--braintrust-no-summary",
        action="store_true",
        default=False,
        help="Suppress the experiment summary at the end of the session.",
    )


def pytest_configure(config: pytest.Config) -> None:
    # Always register the marker so that ``@pytest.mark.braintrust`` never
    # triggers an "unknown marker" warning.
    config.addinivalue_line(
        "markers",
        "braintrust: mark test for Braintrust experiment tracking. "
        "Optional kwargs: project, input, expected, tags, metadata.",
    )

    if config.getoption("--braintrust", default=False):
        import os

        api_key = config.getoption("--braintrust-api-key") or os.environ.get("BRAINTRUST_API_KEY")
        if api_key:
            os.environ["BRAINTRUST_API_KEY"] = api_key

        plugin = BraintrustPytestPlugin(config)
        config.pluginmanager.register(plugin, "braintrust-plugin")


# ---------------------------------------------------------------------------
# braintrust_span fixture (always available)
# ---------------------------------------------------------------------------


@pytest.fixture
def braintrust_span(request: pytest.FixtureRequest) -> BraintrustTestSpan:
    """Return the :class:`BraintrustTestSpan` for the current test.

    When ``--braintrust`` is not active the fixture returns a no-op wrapper
    that silently discards all logged data.
    """
    return getattr(request.node, "_braintrust_test_span", _NOOP_TEST_SPAN)


# ---------------------------------------------------------------------------
# Plugin class (registered only when --braintrust is passed)
# ---------------------------------------------------------------------------


class BraintrustPytestPlugin:
    """Core hook implementation — registered on the pluginmanager only when
    ``--braintrust`` is active."""

    def __init__(self, config: pytest.Config) -> None:
        self._config = config
        self._cli_project: str | None = config.getoption("--braintrust-project")
        self._cli_experiment: str | None = config.getoption("--braintrust-experiment") or None

        # Keyed by experiment group key (module path or project override).
        self.experiments: dict[str, Any] = {}

    # -- helpers ------------------------------------------------------------

    def _get_experiment_key(self, item: pytest.Item) -> str:
        """Determine the experiment grouping key for *item*."""
        # CLI override applies globally.
        if self._cli_project:
            return self._cli_project

        # Per-test / per-class marker override.
        marker = item.get_closest_marker("braintrust")
        if marker and marker.kwargs.get("project"):
            return marker.kwargs["project"]

        # Default: module path.
        return item.module.__name__ if item.module else item.nodeid.split("::")[0]

    def _get_or_create_experiment(self, key: str) -> Any:
        """Get or lazily create an experiment for *key*."""
        if key not in self.experiments:
            import braintrust

            exp = braintrust.init(project=key, experiment=self._cli_experiment)
            self.experiments[key] = exp
        return self.experiments[key]

    def _collect_auto_input(self, item: pytest.Item) -> dict[str, Any] | None:
        """Auto-collect parametrize args as input."""
        if hasattr(item, "callspec"):
            return dict(item.callspec.params)
        return None

    # -- hooks --------------------------------------------------------------

    @pytest.hookimpl(tryfirst=True)
    def pytest_runtest_setup(self, item: pytest.Item) -> None:
        marker = item.get_closest_marker("braintrust")
        if marker is None:
            return

        from braintrust.span_types import SpanTypeAttribute

        key = self._get_experiment_key(item)
        exp = self._get_or_create_experiment(key)

        span = exp.start_span(name=item.name, type=SpanTypeAttribute.EVAL)

        # Static data from the marker.
        marker_input = marker.kwargs.get("input")
        marker_expected = marker.kwargs.get("expected")
        marker_metadata = marker.kwargs.get("metadata")
        marker_tags = marker.kwargs.get("tags")

        # Auto-log parametrize args as input (marker input takes precedence).
        auto_input = self._collect_auto_input(item)
        if marker_input is not None:
            span.log(input=marker_input)
        elif auto_input is not None:
            span.log(input=auto_input)

        if marker_expected is not None:
            span.log(expected=marker_expected)
        if marker_metadata is not None:
            span.log(metadata=marker_metadata)
        if marker_tags is not None:
            span.log(tags=marker_tags)

        test_span = BraintrustTestSpan(span)
        item._braintrust_test_span = test_span  # type: ignore[attr-defined]
        item._braintrust_experiment_key = key  # type: ignore[attr-defined]

        span.set_current()

    @pytest.hookimpl(hookwrapper=True)
    def pytest_runtest_makereport(self, item: pytest.Item, call: pytest.CallInfo) -> None:  # type: ignore[type-arg]
        outcome = yield
        report = outcome.get_result()
        if call.when == "call":
            item._braintrust_report = report  # type: ignore[attr-defined]

    @pytest.hookimpl(trylast=True)
    def pytest_runtest_teardown(self, item: pytest.Item) -> None:
        test_span: BraintrustTestSpan | None = getattr(item, "_braintrust_test_span", None)
        if test_span is None:
            return

        report = getattr(item, "_braintrust_report", None)
        passed = report.passed if report else False

        test_span.span.log(scores={"pass": 1.0 if passed else 0.0})

        if report and report.failed:
            error_str = report.longreprtext if hasattr(report, "longreprtext") else str(report.longrepr)
            test_span.span.log(error=error_str)

        test_span.span.end()
        test_span.span.unset_current()

    def pytest_sessionfinish(self, session: pytest.Session) -> None:
        show_summary = not self._config.getoption("--braintrust-no-summary", default=False)

        for key, exp in self.experiments.items():
            try:
                summary = exp.summarize()
                if show_summary:
                    print(summary)
                exp.flush()
            except Exception:
                traceback.print_exc()
