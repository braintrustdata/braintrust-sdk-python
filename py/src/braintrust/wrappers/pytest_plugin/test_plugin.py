"""Tests for the Braintrust pytest plugin.

Uses the ``pytester`` fixture (built into pytest) to run pytest in-process
with the plugin loaded, combined with mocking ``braintrust.init`` so no real
API calls are made.
"""

from __future__ import annotations

import json
import os
import textwrap

import pytest

# ---------------------------------------------------------------------------
# Tell pytest we need the pytester plugin.
# ---------------------------------------------------------------------------
pytest_plugins = ["pytester"]



# ---------------------------------------------------------------------------
# Helper: inline conftest that mocks braintrust.init for child tests.
#
# The mock experiment captures spans so the parent test can verify data.
# ---------------------------------------------------------------------------

MOCK_CONFTEST = textwrap.dedent(
    '''\
    import json, os, pytest
    from unittest.mock import MagicMock

    _SPANS = []

    class _FakeSpan:
        """Minimal span implementation that records log calls."""
        def __init__(self, name, **kwargs):
            self._data = {"name": name, **{k: v for k, v in kwargs.items() if k != "type"}}
            self._span_type = kwargs.get("type")

        def log(self, **event):
            # Merge scores dicts instead of overwriting so custom + pass coexist.
            if "scores" in event and "scores" in self._data:
                self._data["scores"] = {**self._data["scores"], **event["scores"]}
                rest = {k: v for k, v in event.items() if k != "scores"}
                self._data.update(rest)
            else:
                self._data.update(event)

        def set_current(self):
            pass

        def unset_current(self):
            pass

        def end(self, end_time=None):
            _SPANS.append(self._data.copy())
            return 0

        @property
        def id(self):
            return "fake-span-id"

        @property
        def name(self):
            return self._data.get("name", "")

    _EXPERIMENT = MagicMock()
    _EXPERIMENT.name = "mock-experiment"
    _EXPERIMENT.start_span = lambda name=None, type=None, **kw: _FakeSpan(name=name, type=type, **kw)
    _EXPERIMENT.summarize.return_value = MagicMock(
        experiment_name="mock-experiment",
        scores={},
        metrics={},
        experiment_url=None,
    )

    # Monkey-patch braintrust.init at import time (before any hooks run).
    import braintrust
    braintrust.init = lambda *a, **kw: _EXPERIMENT

    def pytest_sessionfinish(session, exitstatus):
        """Write captured spans to spans.json so the parent test can inspect them."""
        if not _SPANS:
            return
        spans_file = os.path.join(str(session.config.rootpath), "spans.json")
        with open(spans_file, "w") as f:
            json.dump(_SPANS, f, default=str)
    '''
)


# ---------------------------------------------------------------------------
# Helpers for integration tests
# ---------------------------------------------------------------------------


def _run_pytester(pytester: pytest.Pytester, *extra_args: str) -> pytest.RunResult:
    """Run pytester subprocess. The plugin auto-loads via the pytest11 entry point."""
    return pytester.runpytest_subprocess(*extra_args)


def _run_and_get_spans(
    pytester: pytest.Pytester,
    test_code: str,
    extra_args: list[str] | None = None,
) -> list[dict]:
    """Run a child pytest with --braintrust and return captured spans.

    The mock conftest writes spans to ``spans.json`` in the test root dir
    (pytester.path) at session finish.
    """
    spans_file = os.path.join(str(pytester.path), "spans.json")
    pytester.makeconftest(MOCK_CONFTEST)
    pytester.makepyfile(textwrap.dedent(test_code))

    _run_pytester(pytester, "--braintrust", "-s", *(extra_args or []))

    try:
        with open(spans_file) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


# ---------------------------------------------------------------------------
# Integration tests using pytester
# ---------------------------------------------------------------------------


class TestPluginActivation:
    """Tests that the plugin activates / deactivates correctly."""

    def test_no_flag_no_tracking(self, pytester: pytest.Pytester):
        """Without --braintrust, marked tests still run but no tracking occurs."""
        pytester.makeconftest(MOCK_CONFTEST)
        pytester.makepyfile(
            textwrap.dedent(
                """\
                import pytest

                @pytest.mark.braintrust
                def test_example(braintrust_span):
                    # Should get a noop span
                    assert True
                """
            )
        )
        result = _run_pytester(pytester)
        result.assert_outcomes(passed=1)

    def test_flag_enables_tracking(self, pytester: pytest.Pytester):
        """With --braintrust, the plugin activates."""
        pytester.makeconftest(MOCK_CONFTEST)
        pytester.makepyfile(
            textwrap.dedent(
                """\
                import pytest

                @pytest.mark.braintrust
                def test_example(braintrust_span):
                    braintrust_span.log(input={"x": 1})
                    assert True
                """
            )
        )
        result = _run_pytester(pytester, "--braintrust", "-s")
        result.assert_outcomes(passed=1)

    def test_unmarked_tests_unaffected(self, pytester: pytest.Pytester):
        """Tests without the braintrust marker should not be affected."""
        pytester.makeconftest(MOCK_CONFTEST)
        pytester.makepyfile(
            textwrap.dedent(
                """\
                def test_normal():
                    assert 1 + 1 == 2
                """
            )
        )
        result = _run_pytester(pytester, "--braintrust")
        result.assert_outcomes(passed=1)


class TestSpanCreation:
    """Tests that spans are created with correct data."""

    def test_basic_span_data(self, pytester: pytest.Pytester):
        spans = _run_and_get_spans(
            pytester,
            """\
            import pytest

            @pytest.mark.braintrust
            def test_example(braintrust_span):
                braintrust_span.log(input={"query": "hello"}, output={"result": "world"})
                assert True
            """,
        )
        assert len(spans) == 1
        span = spans[0]
        assert span["name"] == "test_example"
        assert span["input"] == {"query": "hello"}
        assert span["output"] == {"result": "world"}
        assert span["scores"] == {"pass": 1.0}

    def test_failing_test_scores_zero(self, pytester: pytest.Pytester):
        spans = _run_and_get_spans(
            pytester,
            """\
            import pytest

            @pytest.mark.braintrust
            def test_fail(braintrust_span):
                assert False, "intentional failure"
            """,
        )
        assert len(spans) == 1
        assert spans[0]["scores"] == {"pass": 0.0}
        assert "error" in spans[0]

    def test_marker_kwargs(self, pytester: pytest.Pytester):
        spans = _run_and_get_spans(
            pytester,
            """\
            import pytest

            @pytest.mark.braintrust(
                input={"query": "hello"},
                expected={"answer": "world"},
                tags=["regression"],
                metadata={"model": "gpt-4"},
            )
            def test_with_kwargs(braintrust_span):
                assert True
            """,
        )
        assert len(spans) == 1
        span = spans[0]
        assert span["input"] == {"query": "hello"}
        assert span["expected"] == {"answer": "world"}
        assert span["tags"] == ["regression"]
        assert span["metadata"] == {"model": "gpt-4"}

    def test_parametrize_auto_input(self, pytester: pytest.Pytester):
        spans = _run_and_get_spans(
            pytester,
            """\
            import pytest

            @pytest.mark.braintrust
            @pytest.mark.parametrize("query,expected_answer", [
                ("2+2?", "4"),
                ("Capital of France?", "Paris"),
            ])
            def test_qa(braintrust_span, query, expected_answer):
                braintrust_span.log(output=expected_answer)
                assert True
            """,
        )
        assert len(spans) == 2
        assert spans[0]["input"] == {"query": "2+2?", "expected_answer": "4"}
        assert spans[1]["input"] == {"query": "Capital of France?", "expected_answer": "Paris"}

    def test_marker_input_overrides_auto_input(self, pytester: pytest.Pytester):
        spans = _run_and_get_spans(
            pytester,
            """\
            import pytest

            @pytest.mark.braintrust(input={"override": True})
            @pytest.mark.parametrize("x", [1, 2])
            def test_override(braintrust_span, x):
                assert True
            """,
        )
        assert len(spans) == 2
        for span in spans:
            assert span["input"] == {"override": True}

    def test_custom_scores(self, pytester: pytest.Pytester):
        spans = _run_and_get_spans(
            pytester,
            """\
            import pytest

            @pytest.mark.braintrust
            def test_scores(braintrust_span):
                braintrust_span.log(scores={"relevance": 0.95, "fluency": 0.8})
                assert True
            """,
        )
        assert len(spans) == 1
        # Custom scores get merged with the pass score.
        assert spans[0]["scores"]["pass"] == 1.0
        assert spans[0]["scores"]["relevance"] == 0.95
        assert spans[0]["scores"]["fluency"] == 0.8


class TestClassLevelMarker:
    """Tests for class-level @pytest.mark.braintrust."""

    def test_class_marker_applies_to_methods(self, pytester: pytest.Pytester):
        spans = _run_and_get_spans(
            pytester,
            """\
            import pytest

            @pytest.mark.braintrust(project="my-project")
            class TestMyLLM:
                def test_case_a(self, braintrust_span):
                    braintrust_span.log(input={"from": "a"})
                    assert True

                def test_case_b(self, braintrust_span):
                    braintrust_span.log(input={"from": "b"})
                    assert True
            """,
        )
        assert len(spans) == 2
        assert spans[0]["input"] == {"from": "a"}
        assert spans[1]["input"] == {"from": "b"}


class TestNoopBehavior:
    """Tests that the noop span works when --braintrust is not passed."""

    def test_noop_fixture(self, pytester: pytest.Pytester):
        pytester.makepyfile(
            textwrap.dedent(
                """\
                import pytest

                @pytest.mark.braintrust
                def test_noop(braintrust_span):
                    # All these calls should succeed without error.
                    braintrust_span.log(
                        input={"x": 1},
                        output={"y": 2},
                        expected={"z": 3},
                        scores={"a": 0.5},
                        metadata={"k": "v"},
                    )
                    assert True
                """
            )
        )
        # Run WITHOUT --braintrust but WITH plugin loaded.
        result = _run_pytester(pytester)
        result.assert_outcomes(passed=1)


class TestSessionFinish:
    """Tests for experiment flush at session end."""

    def test_experiments_flushed(self, pytester: pytest.Pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        pytester.makepyfile(
            textwrap.dedent(
                """\
                import pytest

                @pytest.mark.braintrust
                def test_one(braintrust_span):
                    assert True

                @pytest.mark.braintrust
                def test_two(braintrust_span):
                    assert True
                """
            )
        )
        result = _run_pytester(pytester, "--braintrust", "-s")
        result.assert_outcomes(passed=2)

    def test_summary_suppressed(self, pytester: pytest.Pytester):
        pytester.makeconftest(MOCK_CONFTEST)
        pytester.makepyfile(
            textwrap.dedent(
                """\
                import pytest

                @pytest.mark.braintrust
                def test_one(braintrust_span):
                    assert True
                """
            )
        )
        result = _run_pytester(pytester, "--braintrust", "--braintrust-no-summary", "-s")
        result.assert_outcomes(passed=1)
        # Should not contain experiment summary.
        assert "SUMMARY" not in result.stdout.str()
