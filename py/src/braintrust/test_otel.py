# pylint: disable=not-context-manager
import json
import sys
from collections.abc import Iterator

import pytest
from braintrust.api._test_server import scripted_server


def _check_otel_installed():
    """Check if OpenTelemetry SDK is fully installed."""
    try:
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter  # noqa: F401
        from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor  # noqa: F401

        return True
    except ImportError:
        return False


OTEL_INSTALLED = _check_otel_installed()


_collector_requests: list[tuple[str, dict[str, str]]] = []


def _record_otlp_request(method, path, body, headers):
    _collector_requests.append((path, {key.lower(): value for key, value in headers.items()}))
    return 200, {}, b""


@pytest.fixture
def otlp_collector() -> Iterator[str]:
    """Local OTLP/HTTP endpoint that records the path and headers of each export request."""
    with scripted_server(_record_otlp_request) as (url, _):
        yield url


def _export_span(exporter) -> list[tuple[str, dict[str, str]]]:
    """Export a single span through *exporter* and return the (path, lowercased headers) the collector received."""
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    _collector_requests.clear()
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    provider.get_tracer(__name__).start_span("test-span").end()
    return list(_collector_requests)


def _export_span_and_get_request(exporter) -> tuple[str, dict[str, str]]:
    [request] = _export_span(exporter)
    return request


@pytest.fixture
def uninstall_braintrust_otel():
    sys.modules.pop("braintrust.otel", None)
    yield
    sys.modules.pop("braintrust.otel", None)


def test_otel_import_behavior():
    """Test that OtelExporter can be imported and behaves correctly based on OpenTelemetry availability."""
    from braintrust.otel import OtelExporter

    if _check_otel_installed():
        # Should be able to create an instance with proper params
        assert hasattr(OtelExporter, "__init__")
    else:
        # Import succeeds but instantiation should raise ImportError
        assert hasattr(OtelExporter, "__init__")


def test_otel_exporter_creation(tmp_path):
    """Test OtelExporter creation with and without full OpenTelemetry SDK."""
    from braintrust.otel import OtelExporter

    if _check_otel_installed():
        with pytest.MonkeyPatch.context() as m:
            # Clear any existing environment variables first
            m.delenv("BRAINTRUST_API_KEY", raising=False)
            m.delenv("BRAINTRUST_PARENT", raising=False)

            # Set test environment variables
            m.setenv("BRAINTRUST_API_KEY", "test-api-key")
            m.setenv("BRAINTRUST_PARENT", "project_name:test")

            exporter = OtelExporter()
            assert exporter.parent == "project_name:test"

        with pytest.MonkeyPatch.context() as m:
            m.delenv("BRAINTRUST_API_KEY", raising=False)
            m.delenv("BRAINTRUST_PARENT", raising=False)
            m.chdir(tmp_path)
            (tmp_path / ".env.braintrust").write_text("")

            exporter = OtelExporter()
            with pytest.raises(ValueError, match="API key is required"):
                exporter.force_flush()
    else:
        # When SDK is not fully installed, instantiation should raise ImportError
        with pytest.raises(ImportError, match="OpenTelemetry packages are not installed"):
            OtelExporter(api_key="fake-key")


def test_otel_exporter_with_explicit_params(otlp_collector):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import OtelExporter

    exporter = OtelExporter(
        url=f"{otlp_collector}/custom/v1/traces",
        api_key="explicit-api-key",
        parent="project_name:explicit-test",
        headers={"custom-header": "custom-value"},
    )

    assert exporter.parent == "project_name:explicit-test"

    path, headers = _export_span_and_get_request(exporter)
    assert path == "/custom/v1/traces"
    assert headers["authorization"] == "Bearer explicit-api-key"
    assert headers["x-bt-parent"] == "project_name:explicit-test"
    assert headers["custom-header"] == "custom-value"


def test_otel_exporter_uses_env_braintrust_api_key(tmp_path, otlp_collector):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import OtelExporter

    with pytest.MonkeyPatch.context() as m:
        m.delenv("BRAINTRUST_API_KEY", raising=False)
        m.chdir(tmp_path)
        (tmp_path / ".env.braintrust").write_text("BRAINTRUST_API_KEY=file-api-key\n")

        exporter = OtelExporter(url=f"{otlp_collector}/otel/v1/traces", parent="project_name:test")

        _, headers = _export_span_and_get_request(exporter)
        assert headers["authorization"] == "Bearer file-api-key"
        assert headers["x-bt-parent"] == "project_name:test"


def test_otel_exporter_sends_api_key_resolved_after_construction(otlp_collector):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import OtelExporter

    with pytest.MonkeyPatch.context() as m:
        m.delenv("BRAINTRUST_API_KEY", raising=False)
        exporter = OtelExporter(url=f"{otlp_collector}/otel/v1/traces", parent="project_name:test")

        m.setenv("BRAINTRUST_API_KEY", "late-api-key")
        _, headers = _export_span_and_get_request(exporter)
        assert headers["authorization"] == "Bearer late-api-key"

        # Subsequent exports keep sending the resolved key.
        _, headers = _export_span_and_get_request(exporter)
        assert headers["authorization"] == "Bearer late-api-key"


def test_otel_exporter_resolving_api_key_does_not_revive_shutdown_exporter(otlp_collector):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import OtelExporter

    with pytest.MonkeyPatch.context() as m:
        m.delenv("BRAINTRUST_API_KEY", raising=False)
        exporter = OtelExporter(url=f"{otlp_collector}/otel/v1/traces", parent="project_name:test")
        exporter.shutdown()

        m.setenv("BRAINTRUST_API_KEY", "late-api-key")
        assert _export_span(exporter) == []


def test_otel_exporter_shutdown_racing_lazy_api_key_reinit_stays_shutdown(otlp_collector):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    import threading

    from braintrust.otel import OtelExporter
    from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter

    upstream_init = OTLPSpanExporter.__init__

    def init_with_concurrent_shutdown(self, *args, **kwargs):
        # Run shutdown() on another thread while the lazy-key re-init is in progress.
        # Give it a bounded chance to finish first, so the test also terminates when
        # shutdown correctly waits for the re-init.
        shutdown_thread = threading.Thread(target=self.shutdown)
        shutdown_thread.start()
        shutdown_thread.join(timeout=0.5)
        upstream_init(self, *args, **kwargs)
        shutdown_threads.append(shutdown_thread)

    shutdown_threads = []
    with pytest.MonkeyPatch.context() as m:
        m.delenv("BRAINTRUST_API_KEY", raising=False)
        exporter = OtelExporter(url=f"{otlp_collector}/otel/v1/traces", parent="project_name:test")

        m.setenv("BRAINTRUST_API_KEY", "late-api-key")
        m.setattr(OTLPSpanExporter, "__init__", init_with_concurrent_shutdown)
        _export_span(exporter)
        for shutdown_thread in shutdown_threads:
            shutdown_thread.join(timeout=5)
            assert not shutdown_thread.is_alive()

        assert shutdown_threads
        assert _export_span(exporter) == []


def test_otel_exporter_lazy_api_key_does_not_override_explicit_authorization_header(otlp_collector):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import OtelExporter

    with pytest.MonkeyPatch.context() as m:
        m.delenv("BRAINTRUST_API_KEY", raising=False)
        exporter = OtelExporter(
            url=f"{otlp_collector}/otel/v1/traces",
            parent="project_name:test",
            headers={"Authorization": "Bearer custom-auth"},
        )

        m.setenv("BRAINTRUST_API_KEY", "late-api-key")
        _, headers = _export_span_and_get_request(exporter)
        assert headers["authorization"] == "Bearer custom-auth"


def test_braintrust_span_processor_merges_span_origin_with_context_json_set_after_start():
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import BraintrustSpanProcessor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    processor = BraintrustSpanProcessor(
        api_key="test-api-key",
        parent="project_name:test",
        SpanProcessor=lambda _exporter: SimpleSpanProcessor(memory_exporter),
    )
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test_tracer")

    try:
        with tracer.start_as_current_span("late-context") as span:
            span.set_attribute("braintrust.context_json", json.dumps({"metadata": {"source": "late-attribute"}}))

        provider.force_flush()
        spans = memory_exporter.get_finished_spans()
        assert len(spans) == 1
        context = json.loads(spans[0].attributes["braintrust.context_json"])
        assert context["metadata"]["source"] == "late-attribute"
        assert context["span_origin"]["name"] == "braintrust.sdk.python"
        assert context["span_origin"]["instrumentation"]["name"] == "braintrust-python-otel"
        assert "braintrust.environment.type" not in spans[0].attributes
        assert "braintrust.environment.name" not in spans[0].attributes
    finally:
        provider.shutdown()


def test_braintrust_span_processor_filters_before_adding_span_origin():
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import BraintrustSpanProcessor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    memory_exporter = InMemorySpanExporter()
    provider = TracerProvider()
    processor = BraintrustSpanProcessor(
        api_key="test-api-key",
        parent="project_name:test",
        filter_ai_spans=True,
        SpanProcessor=lambda _exporter: SimpleSpanProcessor(memory_exporter),
    )
    provider.add_span_processor(processor)
    tracer = provider.get_tracer("test_tracer")

    try:
        with tracer.start_as_current_span("GET /health") as span:
            span.set_attribute("http.route", "/health")
        with tracer.start_as_current_span("SELECT users") as span:
            span.set_attribute("db.system", "postgresql")
        with tracer.start_as_current_span("chat") as span:
            span.set_attribute("gen_ai.operation.name", "chat")

        provider.force_flush()
        spans = memory_exporter.get_finished_spans()
        assert [span.name for span in spans] == ["chat"]
        context = json.loads(spans[0].attributes["braintrust.context_json"])
        assert context["span_origin"]["instrumentation"]["name"] == "braintrust-python-otel"
    finally:
        provider.shutdown()


def test_merge_span_origin_context_uses_passed_instrumentation_name():
    from braintrust.span_origin import merge_span_origin_context

    merged = merge_span_origin_context({}, "openai-auto", None)
    assert merged["span_origin"]["instrumentation"] == {"name": "openai-auto"}


def test_span_impl_records_internal_instrumentation():
    from braintrust.logger import (
        Logger,
        ObjectMetadata,
        OrgProjectMetadata,
        _internal_with_memory_background_logger,
    )
    from braintrust.util import LazyValue

    metadata = OrgProjectMetadata(org_id="org", project=ObjectMetadata(id="project", name="project", full_info={}))
    test_logger = Logger(LazyValue(lambda: metadata, use_mutex=False))
    with _internal_with_memory_background_logger() as memory_logger:
        with test_logger.start_span(name="parent", internal={"instrumentation": "openai-auto"}) as parent:
            with parent.start_span(name="child") as child:
                pass
        rows = {row["id"]: row for row in memory_logger.pop()}

    assert rows[parent.id]["context"]["span_origin"]["instrumentation"]["name"] == "openai-auto"
    # Child spans do NOT inherit the parent's instrumentation name;
    # they fall back to the channel default.
    assert rows[child.id]["context"]["span_origin"]["instrumentation"]["name"] == "braintrust-python-logger"


def test_detect_environment_classifies_aws_ecs_before_lambda(monkeypatch):
    from braintrust.span_origin import detect_environment

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITLAB_CI", raising=False)
    monkeypatch.delenv("CIRCLECI", raising=False)
    monkeypatch.delenv("BUILDKITE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("BRAINTRUST_ENVIRONMENT_TYPE", raising=False)
    monkeypatch.delenv("BRAINTRUST_ENVIRONMENT_NAME", raising=False)
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI", raising=False)
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI_V4", raising=False)
    monkeypatch.setenv("AWS_EXECUTION_ENV", "AWS_ECS_FARGATE")

    assert detect_environment() == {"type": "server", "name": "ecs"}


def test_detect_environment_preserves_explicit_name_without_type(monkeypatch):
    from braintrust.span_origin import detect_environment

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITLAB_CI", raising=False)
    monkeypatch.delenv("CIRCLECI", raising=False)
    monkeypatch.delenv("BUILDKITE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("BRAINTRUST_ENVIRONMENT_TYPE", raising=False)
    monkeypatch.setenv("BRAINTRUST_ENVIRONMENT_NAME", "staging")

    assert detect_environment() == {"name": "staging"}


def test_detect_environment_preserves_custom_python_env(monkeypatch):
    from braintrust.span_origin import detect_environment

    for key in (
        "BRAINTRUST_ENVIRONMENT_TYPE",
        "BRAINTRUST_ENVIRONMENT_NAME",
        "GITHUB_ACTIONS",
        "GITLAB_CI",
        "CIRCLECI",
        "BUILDKITE",
        "JENKINS_URL",
        "JENKINS_HOME",
        "TF_BUILD",
        "TEAMCITY_VERSION",
        "TRAVIS",
        "BITBUCKET_BUILD_NUMBER",
        "CI",
        "VERCEL",
        "NETLIFY",
        "ECS_CONTAINER_METADATA_URI",
        "ECS_CONTAINER_METADATA_URI_V4",
        "AWS_EXECUTION_ENV",
        "AWS_LAMBDA_FUNCTION_NAME",
        "K_SERVICE",
        "FUNCTION_TARGET",
        "KUBERNETES_SERVICE_HOST",
        "DYNO",
        "FLY_APP_NAME",
        "RAILWAY_ENVIRONMENT",
        "RENDER_SERVICE_NAME",
    ):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("PYTHON_ENV", "preview")

    assert detect_environment() == {"name": "preview"}


def test_detect_environment_classifies_lambda_when_lambda_specific(monkeypatch):
    from braintrust.span_origin import detect_environment

    monkeypatch.delenv("GITHUB_ACTIONS", raising=False)
    monkeypatch.delenv("GITLAB_CI", raising=False)
    monkeypatch.delenv("CIRCLECI", raising=False)
    monkeypatch.delenv("BUILDKITE", raising=False)
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("BRAINTRUST_ENVIRONMENT_TYPE", raising=False)
    monkeypatch.delenv("BRAINTRUST_ENVIRONMENT_NAME", raising=False)
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI", raising=False)
    monkeypatch.delenv("ECS_CONTAINER_METADATA_URI_V4", raising=False)
    monkeypatch.setenv("AWS_EXECUTION_ENV", "AWS_Lambda_python3.12")

    assert detect_environment() == {"type": "server", "name": "aws_lambda"}


def test_braintrust_span_processor_missing_key_raises_on_span_end(tmp_path):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import BraintrustSpanProcessor
    from opentelemetry.sdk.trace import TracerProvider

    with pytest.MonkeyPatch.context() as m:
        m.delenv("BRAINTRUST_API_KEY", raising=False)
        m.chdir(tmp_path)
        (tmp_path / ".env.braintrust").write_text("")

        provider = TracerProvider()
        processor = BraintrustSpanProcessor()
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test_tracer")

        try:
            with pytest.raises(ValueError, match="API key is required"):
                with tracer.start_as_current_span("test_span"):
                    pass
        finally:
            provider.shutdown()


def test_otel_exporter_no_parent(caplog):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    import logging

    from braintrust.otel import OtelExporter

    with pytest.MonkeyPatch.context() as m:
        m.setenv("BRAINTRUST_API_KEY", "test-api-key")
        m.delenv("BRAINTRUST_PARENT", raising=False)

        # Capture log messages
        with caplog.at_level(logging.INFO):
            exporter = OtelExporter()

        # Check that default parent is set
        assert exporter.parent == "project_name:default-otel-project"

        # Check that logging message is shown
        assert "No parent specified, using default: project_name:default-otel-project" in caplog.text
        assert "Configure with BRAINTRUST_PARENT environment variable or parent parameter" in caplog.text


def test_braintrust_api_url_env_var():
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import OtelExporter

    # Test default URL
    with pytest.MonkeyPatch.context() as m:
        m.setenv("BRAINTRUST_API_KEY", "test-api-key")
        m.setenv("BRAINTRUST_PARENT", "project_name:test")

        exporter = OtelExporter()

        assert exporter._endpoint == "https://api.braintrust.dev/otel/v1/traces"

    # Test custom API URL
    with pytest.MonkeyPatch.context() as m:
        m.setenv("BRAINTRUST_API_KEY", "custom-key")
        m.setenv("BRAINTRUST_API_URL", "https://custom.braintrust.dev")
        m.delenv("BRAINTRUST_PARENT", raising=False)

        exporter = OtelExporter()

        assert exporter._endpoint == "https://custom.braintrust.dev/otel/v1/traces"

    # Test custom API URL with trailing slash
    with pytest.MonkeyPatch.context() as m:
        m.setenv("BRAINTRUST_API_KEY", "custom-key")
        m.setenv("BRAINTRUST_API_URL", "https://custom.example.com/")
        m.delenv("BRAINTRUST_PARENT", raising=False)

        exporter = OtelExporter()

        assert exporter._endpoint == "https://custom.example.com/otel/v1/traces"


def test_otel_exporter_sends_env_api_key_and_default_parent(otlp_collector):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import OtelExporter

    with pytest.MonkeyPatch.context() as m:
        m.setenv("BRAINTRUST_API_KEY", "env-api-key")
        m.setenv("BRAINTRUST_API_URL", otlp_collector)
        m.delenv("BRAINTRUST_PARENT", raising=False)

        exporter = OtelExporter()

        path, headers = _export_span_and_get_request(exporter)
        assert path == "/otel/v1/traces"
        assert headers["authorization"] == "Bearer env-api-key"
        assert headers["x-bt-parent"] == "project_name:default-otel-project"


def test_braintrust_otel_filter_ai_spans_environment_variable():
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    import os

    from braintrust.otel import AISpanProcessor

    # Test that the environment variable is properly read
    original_value = os.environ.get("BRAINTRUST_OTEL_FILTER_AI_SPANS")

    try:
        # Test true value
        os.environ["BRAINTRUST_OTEL_FILTER_AI_SPANS"] = "true"
        assert os.environ.get("BRAINTRUST_OTEL_FILTER_AI_SPANS", "").lower() == "true"

        # Test false value
        os.environ["BRAINTRUST_OTEL_FILTER_AI_SPANS"] = "false"
        assert os.environ.get("BRAINTRUST_OTEL_FILTER_AI_SPANS", "").lower() == "false"

        # Test empty value
        os.environ["BRAINTRUST_OTEL_FILTER_AI_SPANS"] = ""
        assert os.environ.get("BRAINTRUST_OTEL_FILTER_AI_SPANS", "").lower() == ""

        # Test FilterSpanProcessor can be instantiated
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        memory_exporter = InMemorySpanExporter()
        simple_processor = SimpleSpanProcessor(memory_exporter)
        filter_processor = AISpanProcessor(simple_processor)

        # Verify it has the expected attributes
        assert hasattr(filter_processor, "_processor")
        assert hasattr(filter_processor, "_custom_filter")
        assert hasattr(filter_processor, "_on_ending")
        assert hasattr(filter_processor, "_should_keep_filtered_span")
        assert callable(filter_processor._on_ending)
        assert callable(filter_processor._should_keep_filtered_span)

    finally:
        # Restore original value
        if original_value is not None:
            os.environ["BRAINTRUST_OTEL_FILTER_AI_SPANS"] = original_value
        else:
            os.environ.pop("BRAINTRUST_OTEL_FILTER_AI_SPANS", None)


def test_braintrust_span_processor_class(otlp_collector):
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import BraintrustSpanProcessor

    # Test basic processor without filtering
    with pytest.MonkeyPatch.context() as m:
        m.setenv("BRAINTRUST_API_KEY", "test-api-key")
        processor = BraintrustSpanProcessor()

        # Should have the span processor interface
        assert hasattr(processor, "on_start")
        assert hasattr(processor, "on_end")
        assert hasattr(processor, "_on_ending")
        assert hasattr(processor, "shutdown")
        assert hasattr(processor, "force_flush")
        assert callable(processor.on_start)
        assert callable(processor.on_end)
        assert callable(processor._on_ending)
        assert callable(processor.shutdown)
        assert callable(processor.force_flush)

        # Should have access to underlying components
        assert hasattr(processor, "exporter")
        assert hasattr(processor, "processor")

    # Test processor with LLM filtering
    with pytest.MonkeyPatch.context() as m:
        m.setenv("BRAINTRUST_API_KEY", "test-api-key")
        processor_with_filtering = BraintrustSpanProcessor(filter_ai_spans=True)

        # Should have the same interface
        assert hasattr(processor_with_filtering, "on_start")
        assert hasattr(processor_with_filtering, "on_end")
        assert hasattr(processor_with_filtering, "_on_ending")
        assert hasattr(processor_with_filtering, "shutdown")
        assert hasattr(processor_with_filtering, "force_flush")

    # Test processor with custom parameters
    with pytest.MonkeyPatch.context() as m:
        m.setenv("BRAINTRUST_API_KEY", "test-api-key")

        def custom_filter(span):
            return span.name.startswith("test_")

        processor_custom = BraintrustSpanProcessor(
            api_key="explicit-key",
            parent="project:test",
            api_url=otlp_collector,
            filter_ai_spans=True,
            custom_filter=custom_filter,
            headers={"X-Test-Header": "test"},
        )

        # Should have the same interface
        assert hasattr(processor_custom, "on_start")
        assert hasattr(processor_custom, "on_end")
        assert hasattr(processor_custom, "shutdown")
        assert hasattr(processor_custom, "force_flush")

        # Check that the exporter was created with the right parameters
        exporter = processor_custom.exporter
        assert exporter.parent == "project:test"
        path, headers = _export_span_and_get_request(exporter)
        assert path == "/otel/v1/traces"
        assert headers["authorization"] == "Bearer explicit-key"
        assert headers["x-bt-parent"] == "project:test"
        assert headers["x-test-header"] == "test"


class TestSpanFiltering:
    def setup_method(self):
        try:
            from opentelemetry.sdk.trace import TracerProvider  # noqa: F401
        except ImportError:
            pytest.skip("OpenTelemetry SDK not fully installed, skipping AISpanProcessor tests")

        from braintrust.otel import AISpanProcessor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        self.memory_exporter = InMemorySpanExporter()
        self.provider = TracerProvider()

        # Create processor with our filtering logic
        base_processor = SimpleSpanProcessor(self.memory_exporter)
        self.filtering_processor = AISpanProcessor(base_processor)

        self.provider.add_span_processor(self.filtering_processor)
        self.tracer = self.provider.get_tracer("test_tracer")

    def teardown_method(self):
        if OTEL_INSTALLED:
            self.provider.shutdown()
            self.memory_exporter.clear()

    def test_filters_out_root_spans(self):
        with self.tracer.start_as_current_span("root_operation"):
            pass

        spans = self.memory_exporter.get_finished_spans()
        assert len(spans) == 0

    def test_keeps_gen_ai_spans(self):
        with self.tracer.start_as_current_span("root"):
            with self.tracer.start_as_current_span("gen_ai.completion"):
                pass
            with self.tracer.start_as_current_span("regular_operation"):
                pass

        spans = self.memory_exporter.get_finished_spans()
        span_names = [span.name for span in spans]

        assert "root" not in span_names
        assert "gen_ai.completion" in span_names
        assert "regular_operation" not in span_names

    def test_keeps_braintrust_spans(self):
        with self.tracer.start_as_current_span("root"):
            with self.tracer.start_as_current_span("braintrust.eval"):
                pass
            with self.tracer.start_as_current_span("database_query"):
                pass

        spans = self.memory_exporter.get_finished_spans()
        span_names = [span.name for span in spans]

        assert "braintrust.eval" in span_names
        assert "database_query" not in span_names

    def test_keeps_traceloop_spans(self):
        with self.tracer.start_as_current_span("root"):
            with self.tracer.start_as_current_span("traceloop.agent"):
                pass
            with self.tracer.start_as_current_span("traceloop.workflow.step"):
                pass

        spans = self.memory_exporter.get_finished_spans()
        span_names = [span.name for span in spans]
        assert "root" not in span_names
        assert "traceloop.agent" in span_names
        assert "traceloop.workflow.step" in span_names

    def test_keeps_llm_spans(self):
        with self.tracer.start_as_current_span("root"):
            with self.tracer.start_as_current_span("llm.generate"):
                pass

        spans = self.memory_exporter.get_finished_spans()
        span_names = [span.name for span in spans]
        assert "llm.generate" in span_names

    def test_keeps_ai_spans(self):
        with self.tracer.start_as_current_span("root"):
            with self.tracer.start_as_current_span("ai.model_call"):
                pass

        spans = self.memory_exporter.get_finished_spans()
        span_names = [span.name for span in spans]
        assert "root" not in span_names
        assert "ai.model_call" in span_names

    def test_keeps_spans_with_llm_attributes(self):
        with self.tracer.start_as_current_span("root"):
            with self.tracer.start_as_current_span("some_operation") as span:
                span.set_attribute("gen_ai.model", "gpt-4")
                span.set_attribute("regular_data", "value")
            with self.tracer.start_as_current_span("another_operation") as span:
                span.set_attribute("llm.tokens", 100)
            with self.tracer.start_as_current_span("traceloop_operation") as span:
                span.set_attribute("traceloop.agent_id", "agent-123")
            with self.tracer.start_as_current_span("third_operation") as span:
                span.set_attribute("database.connection", "postgres")

        spans = self.memory_exporter.get_finished_spans()
        span_names = [span.name for span in spans]

        assert "root" not in span_names
        assert "some_operation" in span_names  # has gen_ai.model attribute
        assert "another_operation" in span_names  # has llm.tokens attribute
        assert "traceloop_operation" in span_names  # has traceloop.agent_id attribute
        assert "third_operation" not in span_names  # no LLM attributes

    def test_drops_non_llm_spans(self):
        with self.tracer.start_as_current_span("root"):
            with self.tracer.start_as_current_span("database_query"):
                pass
            with self.tracer.start_as_current_span("http_request"):
                pass
            with self.tracer.start_as_current_span("file_operation"):
                pass

        spans = self.memory_exporter.get_finished_spans()
        assert len(spans) == 0

    def test_custom_filter_keeps_spans(self):
        def custom_filter(span):
            if span.name == "custom_keep":
                return True
            return None  # Don't influence decision

        # Create processor with custom filter
        from braintrust.otel import AISpanProcessor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        memory_exporter = InMemorySpanExporter()
        processor = AISpanProcessor(SimpleSpanProcessor(memory_exporter), custom_filter=custom_filter)
        provider = TracerProvider()
        provider.add_span_processor(processor)
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("root"):
            with tracer.start_as_current_span("custom_keep"):
                pass
            with tracer.start_as_current_span("regular_operation"):
                pass

        spans = memory_exporter.get_finished_spans()
        span_names = [span.name for span in spans]

        assert "custom_keep" in span_names  # kept by custom filter
        assert "regular_operation" not in span_names  # dropped by default logic
        assert "root" not in span_names

    def test_custom_filter_drops_spans(self):
        def custom_filter(span):
            if span.name == "gen_ai.drop_this":
                return False
            return None  # Don't influence decision

        # Create processor with custom filter
        from braintrust.otel import AISpanProcessor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        memory_exporter = InMemorySpanExporter()
        processor = AISpanProcessor(SimpleSpanProcessor(memory_exporter), custom_filter=custom_filter)
        provider = TracerProvider()
        provider.add_span_processor(processor)
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("root"):
            with tracer.start_as_current_span("gen_ai.drop_this"):
                pass
            with tracer.start_as_current_span("gen_ai.keep_this"):
                pass

        spans = memory_exporter.get_finished_spans()
        span_names = [span.name for span in spans]

        assert "gen_ai.drop_this" not in span_names  # dropped by custom filter
        assert "gen_ai.keep_this" in span_names  # kept by default LLM logic
        assert "root" not in span_names

    def test_custom_filter_none_uses_default_logic(self):
        def custom_filter(span):
            return None  # Always defer to default logic

        # Create processor with custom filter
        from braintrust.otel import AISpanProcessor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        memory_exporter = InMemorySpanExporter()
        processor = AISpanProcessor(SimpleSpanProcessor(memory_exporter), custom_filter=custom_filter)
        provider = TracerProvider()
        provider.add_span_processor(processor)
        tracer = provider.get_tracer(__name__)

        with tracer.start_as_current_span("root"):
            with tracer.start_as_current_span("gen_ai.completion"):
                pass
            with tracer.start_as_current_span("regular_operation"):
                pass

        spans = memory_exporter.get_finished_spans()
        span_names = [span.name for span in spans]

        assert "root" not in span_names
        assert "gen_ai.completion" in span_names  # kept by default LLM logic
        assert "regular_operation" not in span_names  # dropped by default logic

    def test_filtering_vs_unfiltered_comparison(self):
        # Set up two separate exporters and processors
        from braintrust.otel import AISpanProcessor
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        all_spans_exporter = InMemorySpanExporter()
        filtered_spans_exporter = InMemorySpanExporter()

        # Processor that captures everything
        all_processor = SimpleSpanProcessor(all_spans_exporter)

        # Processor that filters LLM spans
        filtered_processor = AISpanProcessor(SimpleSpanProcessor(filtered_spans_exporter))

        # Set up provider with both processors
        provider = TracerProvider()
        provider.add_span_processor(all_processor)
        provider.add_span_processor(filtered_processor)
        tracer = provider.get_tracer("comparison_test")

        # Create a mix of spans - some LLM-related, some not
        with tracer.start_as_current_span("user_request") as root:
            root.set_attribute("request.id", "123")

            with tracer.start_as_current_span("database_query"):
                pass

            with tracer.start_as_current_span("gen_ai.completion") as llm_span:
                llm_span.set_attribute("gen_ai.model", "gpt-4")

            with tracer.start_as_current_span("cache_lookup"):
                pass

            with tracer.start_as_current_span("response_formatting") as resp_span:
                resp_span.set_attribute("llm.tokens", 150)

            with tracer.start_as_current_span("http_response"):
                pass

        # Clean up
        provider.shutdown()

        # Verify all spans were captured by the unfiltered exporter
        all_spans = all_spans_exporter.get_finished_spans()
        all_span_names = [span.name for span in all_spans]

        assert len(all_spans) == 6
        assert "user_request" in all_span_names
        assert "database_query" in all_span_names
        assert "gen_ai.completion" in all_span_names
        assert "cache_lookup" in all_span_names
        assert "response_formatting" in all_span_names
        assert "http_response" in all_span_names

        # Verify only LLM spans were captured by the filtered exporter
        filtered_spans = filtered_spans_exporter.get_finished_spans()
        filtered_span_names = [span.name for span in filtered_spans]

        assert len(filtered_spans) == 2
        assert "user_request" not in filtered_span_names  # root span
        assert "gen_ai.completion" in filtered_span_names  # LLM name
        assert "response_formatting" in filtered_span_names  # LLM attribute

    def test_custom_filter_is_root_span(self):
        from braintrust.otel import AISpanProcessor, is_root_span
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import SimpleSpanProcessor
        from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

        memory_exporter = InMemorySpanExporter()
        processor = AISpanProcessor(SimpleSpanProcessor(memory_exporter), custom_filter=is_root_span)
        provider = TracerProvider()
        provider.add_span_processor(processor)
        tracer = provider.get_tracer("test-braintrust-root-filter")

        with tracer.start_as_current_span("root_span"):
            with tracer.start_as_current_span("child_span"):
                pass

        provider.shutdown()
        spans = memory_exporter.get_finished_spans()
        names = [span.name for span in spans]
        assert "root_span" in names
        assert "child_span" not in names


def test_parent_from_headers_invalid_inputs():
    """Test parent_from_headers with various invalid inputs."""
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import parent_from_headers

    # Test 1: Empty headers
    result = parent_from_headers({})
    assert result is None

    # Test 2: Invalid traceparent (malformed)
    result = parent_from_headers({"traceparent": "invalid"})
    assert result is None

    # Test 3: Valid traceparent but invalid braintrust.parent format
    result = parent_from_headers(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "baggage": "braintrust.parent=invalid_format",
        }
    )
    assert result is None

    # Test 4: Empty project_id
    result = parent_from_headers(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "baggage": "braintrust.parent=project_id:",
        }
    )
    assert result is None

    # Test 5: Empty project_name
    result = parent_from_headers(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "baggage": "braintrust.parent=project_name:",
        }
    )
    assert result is None

    # Test 6: Empty experiment_id
    result = parent_from_headers(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "baggage": "braintrust.parent=experiment_id:",
        }
    )
    assert result is None

    # Test 7: Invalid trace_id length (too short)
    result = parent_from_headers(
        {"traceparent": "00-4bf92f3577b34da6-00f067aa0ba902b7-01", "baggage": "braintrust.parent=project_name:test"}
    )
    assert result is None

    # Test 8: Invalid span_id length (too short)
    result = parent_from_headers(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa-01",
            "baggage": "braintrust.parent=project_name:test",
        }
    )
    assert result is None


def test_parent_from_headers_valid_input():
    """Test parent_from_headers with valid inputs."""
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import parent_from_headers

    # Test with valid project_name
    result = parent_from_headers(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "baggage": "braintrust.parent=project_name:test-project",
        }
    )
    assert result is not None
    # Result is base64 encoded, so just check it's a non-empty string
    assert isinstance(result, str)
    assert len(result) > 0

    # Test with valid project_id
    result = parent_from_headers(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "baggage": "braintrust.parent=project_id:abc123",
        }
    )
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0

    # Test with valid experiment_id
    result = parent_from_headers(
        {
            "traceparent": "00-4bf92f3577b34da6a3ce929d0e0e4736-00f067aa0ba902b7-01",
            "baggage": "braintrust.parent=experiment_id:exp-456",
        }
    )
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


def test_add_parent_to_baggage():
    """Test add_parent_to_baggage function."""
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import add_parent_to_baggage
    from opentelemetry import baggage, context

    # Test adding parent to baggage
    token = add_parent_to_baggage("project_name:test-project")
    assert token is not None

    # Verify it's in baggage
    parent_value = baggage.get_baggage("braintrust.parent")
    assert parent_value == "project_name:test-project"

    # Clean up
    context.detach(token)


def test_add_span_parent_to_baggage():
    """Test add_span_parent_to_baggage function."""
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import add_span_parent_to_baggage
    from opentelemetry import baggage, context, trace
    from opentelemetry.sdk.trace import TracerProvider

    # Setup tracer
    provider = TracerProvider()
    trace.set_tracer_provider(provider)
    tracer = trace.get_tracer(__name__)

    # Test with span that has braintrust.parent attribute
    with tracer.start_as_current_span("test_span") as span:
        span.set_attribute("braintrust.parent", "project_name:test")

        token = add_span_parent_to_baggage(span)
        assert token is not None

        # Verify it's in baggage
        parent_value = baggage.get_baggage("braintrust.parent")
        assert parent_value == "project_name:test"

        context.detach(token)

    # Test with span that doesn't have braintrust.parent attribute (should return None and warn)
    with tracer.start_as_current_span("test_span_no_attr") as span:
        token = add_span_parent_to_baggage(span)
        assert token is None

    # Test with None span (should return None and warn)
    token = add_span_parent_to_baggage(None)
    assert token is None


def test_parent_from_headers_with_custom_propagator():
    """Test parent_from_headers with a custom propagator."""
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.otel import parent_from_headers
    from opentelemetry import baggage as otel_baggage
    from opentelemetry import context as otel_context
    from opentelemetry import trace
    from opentelemetry.propagators.textmap import CarrierT, Getter, TextMapPropagator, default_getter
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    class CustomHeaderPropagator(TextMapPropagator):
        """Custom propagator that reads trace context from X-Custom-* headers."""

        def extract(
            self,
            carrier: CarrierT,
            context: otel_context.Context | None = None,
            getter: Getter = default_getter,
        ) -> otel_context.Context:
            if context is None:
                context = otel_context.get_current()

            trace_id = getter.get(carrier, "X-Custom-Trace-Id")
            span_id = getter.get(carrier, "X-Custom-Span-Id")

            if trace_id and span_id:
                trace_id_list = trace_id if isinstance(trace_id, list) else [trace_id]
                span_id_list = span_id if isinstance(span_id, list) else [span_id]

                span_context = SpanContext(
                    trace_id=int(trace_id_list[0], 16),
                    span_id=int(span_id_list[0], 16),
                    is_remote=True,
                    trace_flags=TraceFlags.SAMPLED,
                )
                span = NonRecordingSpan(span_context)
                context = trace.set_span_in_context(span, context)

            # Also extract baggage from standard baggage header
            baggage_header = getter.get(carrier, "baggage")
            if baggage_header:
                baggage_list = baggage_header if isinstance(baggage_header, list) else [baggage_header]
                for item in baggage_list[0].split(","):
                    if "=" in item:
                        key, value = item.split("=", 1)
                        context = otel_baggage.set_baggage(key.strip(), value.strip(), context)

            return context

        def inject(self, carrier, context=None, setter=None):
            pass  # Not needed for this test

        @property
        def fields(self):
            return {"X-Custom-Trace-Id", "X-Custom-Span-Id", "baggage"}

    propagator = CustomHeaderPropagator()

    # Custom header format
    headers = {
        "X-Custom-Trace-Id": "4bf92f3577b34da6a3ce929d0e0e4736",
        "X-Custom-Span-Id": "00f067aa0ba902b7",
        "baggage": "braintrust.parent=project_name:test-project",
    }

    result = parent_from_headers(headers, propagator=propagator)
    assert result is not None
    assert isinstance(result, str)
    assert len(result) > 0


# --------------------------------------------------------------------------- #
# Cross-paradigm: native W3C propagation <-> OpenTelemetry propagation
# --------------------------------------------------------------------------- #


def test_native_inject_extracted_by_otel():
    """Native Span.inject() headers are parseable by OTel's W3C propagator."""
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.logger import _internal_with_memory_background_logger
    from braintrust.test_helpers import init_test_logger
    from opentelemetry import trace
    from opentelemetry.propagate import extract

    with _internal_with_memory_background_logger():
        logger = init_test_logger("cross-paradigm-native-to-otel")
        with logger.start_span(name="native_a") as span:
            carrier = span.inject({})
            native_trace_id = span.root_span_id
            native_span_id = span.span_id

    # OTel extracts trace identity from the native-produced W3C headers.
    ctx = extract(carrier)
    otel_span = trace.get_current_span(ctx)
    sc = otel_span.get_span_context()
    assert format(sc.trace_id, "032x") == native_trace_id
    assert format(sc.span_id, "016x") == native_span_id


def test_otel_inject_extracted_by_native():
    """OTel-injected W3C headers (+ braintrust.parent baggage) are parseable natively."""
    if not _check_otel_installed():
        pytest.skip("OpenTelemetry SDK not fully installed, skipping test")

    from braintrust.logger import _internal_with_memory_background_logger, extract_trace_context
    from braintrust.test_helpers import init_test_logger
    from opentelemetry import baggage, context, trace
    from opentelemetry.propagate import inject
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags

    trace_id = 0x4BF92F3577B34DA6A3CE929D0E0E4736
    span_id = 0x00F067AA0BA902B7
    span_context = SpanContext(
        trace_id=trace_id,
        span_id=span_id,
        is_remote=False,
        trace_flags=TraceFlags(TraceFlags.SAMPLED),
    )
    ctx = trace.set_span_in_context(NonRecordingSpan(span_context))
    ctx = baggage.set_baggage("braintrust.parent", "project_id:abc123", context=ctx)

    token = context.attach(ctx)
    try:
        carrier = {}
        inject(carrier)
    finally:
        context.detach(token)

    parent = extract_trace_context(carrier)
    assert parent is not None
    with _internal_with_memory_background_logger():
        logger = init_test_logger("otel-to-native")
        with logger.start_span(name="native", parent=parent) as span:
            assert span.root_span_id == format(trace_id, "032x")
            assert span.span_parents == [format(span_id, "016x")]


@pytest.fixture
def customized_otlp_receiver(monkeypatch):
    if not OTEL_INSTALLED:
        pytest.skip("OpenTelemetry SDK not installed")
    from braintrust import set_span_customizers
    from google.protobuf import symbol_database
    from opentelemetry.proto.collector.trace.v1 import trace_service_pb2

    request_type = symbol_database.Default().GetSymbol(
        f"{trace_service_pb2.DESCRIPTOR.package}.ExportTraceServiceRequest"
    )

    monkeypatch.setattr("braintrust.span_customizer._failure_logged", False)
    requests = []

    def receive(method, path, body, headers):
        request = request_type()
        request.ParseFromString(body)
        requests.append(request)
        return 200, {}, b""

    set_span_customizers(None)
    try:
        with scripted_server(receive) as (url, _):
            yield url, requests
    finally:
        set_span_customizers(None)


def _wire_spans(requests):
    return [
        (resource, scope, span)
        for request in requests
        for resource in request.resource_spans
        for scope in resource.scope_spans
        for span in scope.spans
    ]


def _sensitive_otel_spans():
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import Link, Status, StatusCode

    memory = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({"private.resource": "secret"}), shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("private.scope", "private.version", schema_url="private.schema")
    with tracer.start_as_current_span("private.parent") as parent:
        with tracer.start_as_current_span(
            "private.child", links=[Link(parent.get_span_context(), {"private.link": "secret"})]
        ) as child:
            child.set_attributes(
                {
                    "braintrust.parent": "experiment_id:destination",
                    "braintrust.input_json": '"private.input"',
                    "private.array": ["first", "second"],
                }
            )
            child.add_event("private.event", {"private.event": "secret"}, timestamp=123)
            child.set_status(Status(StatusCode.ERROR, "private.status"))
    spans = memory.get_finished_spans()
    provider.shutdown()
    return spans


@pytest.mark.parametrize("replacement", [False, True])
def test_otel_customizers_real_http_snapshot_replacement_and_identity(customized_otlp_receiver, replacement):
    from braintrust import SpanCustomizer, set_span_customizers
    from braintrust.otel import OtelExporter

    url, requests = customized_otlp_receiver
    spans = _sensitive_otel_spans()
    seen = []

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            assert type(data) is dict
            # Direct exporters don't add provenance, with or without customizers.
            assert "braintrust.context_json" not in data["attributes"]
            seen.append(data["name"])
            data["context"]["span_id"] = 1
            data["context"]["trace_id"] = 2
            if data["parent"]:
                data["parent"]["span_id"] = 3
            data["start_time"] = 4
            data["end_time"] = 5
            data["attributes"]["braintrust.parent"] = "project_name:wrong"
            if "private.array" in data["attributes"]:
                data["attributes"]["private.array"][0] = "safe"
                data["events"][0]["attributes"].clear()
                data["links"][0]["attributes"].clear()
            data["resource"]["attributes"].clear()
            data["instrumentation_scope"]["name"] = "safe.scope"
            if replacement:
                return {"name": "redacted", "attributes": {"safe": True}}
            data["name"] = "redacted"
            data["attributes"] = {"safe": True}
            data["events"] = []
            data["links"] = []
            data.pop("status")
            data.pop("resource")
            data.pop("instrumentation_scope")
            return data

    class Observe(SpanCustomizer):
        def on_span_export(self, data):
            original = spans[len(observed)]
            assert data["context"]["span_id"] == original.context.span_id
            assert data["start_time"] == original.start_time
            observed.append(data)
            return data

    observed = []
    set_span_customizers([Redact(), Observe()])
    exporter = OtelExporter(url=url, api_key="test")
    try:
        exporter.export(spans)
    finally:
        exporter.shutdown()
    assert seen == ["private.child", "private.parent"]
    assert len(observed) == 2
    for (resource, scope, wire), original in zip(_wire_spans(requests), spans):
        assert wire.name == "redacted"
        assert wire.span_id == original.context.span_id.to_bytes(8, "big")
        assert wire.trace_id == original.context.trace_id.to_bytes(16, "big")
        assert wire.parent_span_id == (original.parent.span_id.to_bytes(8, "big") if original.parent else b"")
        assert (wire.start_time_unix_nano, wire.end_time_unix_nano) == (original.start_time, original.end_time)
        assert not wire.events and not wire.links and not wire.status.message
        assert not resource.resource.attributes and not scope.scope.name
        attrs = {attr.key: attr.value for attr in wire.attributes}
        assert attrs["safe"].bool_value
        assert "braintrust.context_json" not in attrs
        if original.parent:
            assert attrs["braintrust.parent"].string_value == "experiment_id:destination"
    child = spans[0]
    assert child.name == "private.child"
    assert child.attributes["private.array"] == ("first", "second")
    assert child.events[0].attributes["private.event"] == "secret"
    assert child.links[0].attributes["private.link"] == "secret"
    assert child.resource.attributes["private.resource"] == "secret"
    assert child.instrumentation_scope.name == "private.scope"


@pytest.mark.parametrize(
    "failure",
    [
        "exception",
        "base_exception",
        "none",
        "coroutine",
        "attributes",
        "attribute_value",
        "integer_overflow",
        "invalid_utf8_key",
        "event",
        "link",
        "null_link_context",
        "status",
        "resource",
        "scope",
        "link_trace_state",
        "name",
        "cycle",
    ],
)
def test_otel_customizer_fail_closed_real_http_mixed_batch(customized_otlp_receiver, caplog, failure):
    import logging

    from braintrust import SpanCustomizer, set_span_customizers
    from braintrust.otel import OtelExporter
    from braintrust.span_customizer import _customize_span_export

    url, requests = customized_otlp_receiver
    spans = _sensitive_otel_spans()
    calls = []

    async def asynchronous():
        return {}

    class Break(SpanCustomizer):
        def on_span_export(self, data):
            if data.get("name") == "private.parent":
                return {"name": "healthy"}
            data["name"] = "private.partial"
            if failure == "exception":
                raise ValueError("private.exception")
            if failure == "base_exception":
                raise GeneratorExit("private.exception")
            if failure == "none":
                return None
            if failure == "coroutine":
                return asynchronous()
            if failure == "cycle":
                data["attributes"] = data
            else:
                key, value = {
                    "attributes": ("attributes", []),
                    "attribute_value": ("attributes", {"private": object()}),
                    "integer_overflow": ("attributes", {"private": 2**64}),
                    "invalid_utf8_key": ("attributes", {"private\ud800": "secret"}),
                    "event": ("events", [{"name": "private", "timestamp": -1}]),
                    "link": ("links", [{"context": {"trace_id": -1, "span_id": 1}}]),
                    "null_link_context": ("links", [{"context": None}]),
                    "status": ("status", {"status_code": 999, "description": "private"}),
                    "resource": ("resource", {"attributes": {"private": object()}}),
                    "scope": ("instrumentation_scope", {"attributes": {"private": object()}}),
                    "link_trace_state": (
                        "links",
                        [{"context": {"trace_id": 1, "span_id": 1, "trace_state": {"private": "secret,"}}}],
                    ),
                    "name": ("name", {}),
                }[failure]
                data[key] = value
            return data

    class Observe(SpanCustomizer):
        def on_span_export(self, data):
            calls.append(data["name"])
            return data

    set_span_customizers([Break(), Observe()])
    exporter = OtelExporter(url=url, api_key="test")
    with caplog.at_level(logging.ERROR):
        try:
            exporter.export(spans)
            exporter.export(spans)
            # Native and OTel share the same once-only diagnostic budget.
            _customize_span_export({"id": "row"}, [Break()])
        finally:
            exporter.shutdown()
    assert calls == ["healthy", "healthy"]
    assert len(caplog.records) == 1
    assert "private" not in caplog.text
    assert len(requests) == 2
    for request in requests:
        assert len(request.resource_spans) == 1
        encoded = request.SerializeToString()
        assert b"private" not in encoded and b"secret" not in encoded
        records = _wire_spans([request])
        assert len(records) == 2
        for resource, scope, wire in records:
            assert not resource.resource.attributes and not resource.schema_url
            assert not scope.scope.name and not scope.scope.version and not scope.schema_url
            assert not wire.events and not wire.links and not wire.status.message
            if wire.name == "healthy":
                continue
            assert not wire.name and not wire.trace_state
            assert wire.span_id == spans[0].context.span_id.to_bytes(8, "big")
            assert wire.trace_id == spans[0].context.trace_id.to_bytes(16, "big")
            assert wire.parent_span_id == spans[0].parent.span_id.to_bytes(8, "big")
            assert (wire.start_time_unix_nano, wire.end_time_unix_nano) == (spans[0].start_time, spans[0].end_time)
            assert {a.key: a.value.string_value for a in wire.attributes} == {
                "braintrust.parent": "experiment_id:destination",
                "braintrust.context_json": '{"customizer_error":true}',
            }


@pytest.mark.parametrize("compat", [False, True])
def test_otel_processor_customizes_after_filter_once_without_mutating_other_exporters(
    customized_otlp_receiver, monkeypatch, compat
):
    from braintrust import SpanCustomizer, set_span_customizers
    from braintrust.otel import BraintrustSpanProcessor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", str(compat).lower())
    url, requests = customized_otlp_receiver
    seen = []

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            seen.append(data["name"])
            assert "span_origin" in json.loads(data["attributes"]["braintrust.context_json"])
            return {"name": "redacted"}

    set_span_customizers([Redact()])
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(
        BraintrustSpanProcessor(
            api_key="test",
            api_url=url,
            SpanProcessor=SimpleSpanProcessor,
            filter_ai_spans=True,
        )
    )
    other = InMemorySpanExporter()
    provider.add_span_processor(SimpleSpanProcessor(other))
    tracer = provider.get_tracer(__name__)
    with tracer.start_as_current_span("gen_ai.root"):
        tracer.start_span("gen_ai.child").end()
        tracer.start_span("filtered").end()
    provider.shutdown()
    assert seen == ["gen_ai.child", "gen_ai.root"]
    assert [wire.name for _, _, wire in _wire_spans(requests)] == ["redacted", "redacted"]
    assert [span.name for span in other.get_finished_spans()] == ["gen_ai.child", "filtered", "gen_ai.root"]


def test_otel_customizer_preserves_successful_snapshot_content(customized_otlp_receiver):
    from braintrust import SpanCustomizer, set_span_customizers
    from braintrust.otel import OtelExporter

    url, requests = customized_otlp_receiver
    spans = _sensitive_otel_spans()

    class Customize(SpanCustomizer):
        def on_span_export(self, data):
            if data["events"]:
                data["events"][0]["name"] = "safe.event"
                data["events"][0]["attributes"] = {"safe.event": True}
                data["links"][0]["attributes"] = {"safe.link": True}
                data["status"]["description"] = "safe.status"
            data["resource"]["attributes"] = {"safe.resource": "value"}
            data["instrumentation_scope"]["name"] = "safe.scope"
            return data

    set_span_customizers([Customize()])
    exporter = OtelExporter(url=url, api_key="test")
    try:
        exporter.export(spans)
    finally:
        exporter.shutdown()
    resource, scope, child = _wire_spans(requests)[0]
    assert child.name == "private.child"
    assert child.events[0].name == "safe.event" and child.events[0].time_unix_nano == 123
    assert child.events[0].attributes[0].value.bool_value
    assert child.links[0].attributes[0].value.bool_value
    assert child.status.message == "safe.status"
    assert resource.resource.attributes[0].key == "safe.resource"
    assert scope.scope.name == "safe.scope"


@pytest.mark.parametrize("experiment", [False, True])
def test_customizers_cover_native_and_otel_compat_roots_and_children(
    customized_otlp_receiver, monkeypatch, experiment
):
    from braintrust import SpanCustomizer, set_span_customizers
    from braintrust.logger import _internal_with_memory_background_logger
    from braintrust.otel import BraintrustSpanProcessor
    from braintrust.test_helpers import init_test_exp, init_test_logger
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor

    monkeypatch.setenv("BRAINTRUST_OTEL_COMPAT", "true")
    url, requests = customized_otlp_receiver
    shapes = set()

    class Redact(SpanCustomizer):
        def on_span_export(self, data):
            if "attributes" in data:
                shapes.add("otel")
                data["attributes"].pop("braintrust.input_json", None)
            else:
                shapes.add("native")
                data.pop("input", None)
            return data

    set_span_customizers([Redact()])
    provider = TracerProvider(shutdown_on_exit=False)
    provider.add_span_processor(
        BraintrustSpanProcessor(api_key="test", api_url=url, SpanProcessor=SimpleSpanProcessor)
    )
    tracer = provider.get_tracer(__name__)
    try:
        with _internal_with_memory_background_logger() as memory:
            owner = init_test_exp("customized", "project") if experiment else init_test_logger("customized")
            with owner.start_span(name="native.root", input="private") as root:
                with root.start_span(name="native.child", input="private"):
                    with tracer.start_as_current_span("otel.root", attributes={"braintrust.input_json": '"private"'}):
                        tracer.start_span("otel.child", attributes={"braintrust.input_json": '"private"'}).end()
            rows = memory.pop()
    finally:
        provider.shutdown()
    assert shapes == {"native", "otel"}
    assert len(rows) == 2 and all("input" not in row for row in rows)
    wires = _wire_spans(requests)
    assert len(wires) == 2
    assert all("braintrust.input_json" not in {a.key for a in wire.attributes} for _, _, wire in wires)


def test_otel_noop_customizer_exports_same_request_as_no_customizers(customized_otlp_receiver):
    from http import HTTPStatus

    from braintrust import SpanCustomizer, set_span_customizers
    from braintrust.otel import OtelExporter
    from opentelemetry import trace as otel_trace
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.trace import NonRecordingSpan, SpanContext, TraceFlags, TraceState

    url, requests = customized_otlp_receiver

    class Label(str):
        pass

    class Measure(float):
        pass

    class Binary(bytes):
        pass

    # Newer SDKs keep None, bytes, mappings, and mixed arrays. Older SDKs keep None
    # inside arrays, which their encoders can't send and skip without customizers.
    attributes = {
        "none": None,
        "bytes": b"value",
        "mapping": {"nested": {"value": 1}},
        "mixed": [1, "two"],
        "none_in_array": ["one", None],
        "string": "value",
        "http.status_code": HTTPStatus.OK,
        "label": Label("value"),
        "measure": Measure(1.5),
        "binary_subclass": Binary(b"value"),
        "status_array": [HTTPStatus.OK, HTTPStatus.NOT_FOUND],
    }

    memory = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({"service.name": "svc", "mapping": {"a": 1}}), shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("scope", "1.0")
    remote = SpanContext(
        trace_id=1, span_id=2, is_remote=True, trace_flags=TraceFlags(1), trace_state=TraceState([("k", "v")])
    )
    with tracer.start_as_current_span("root", context=otel_trace.set_span_in_context(NonRecordingSpan(remote))):
        with tracer.start_as_current_span("child") as child:
            for key, value in attributes.items():
                child.set_attribute(key, value)
            child.add_event("event", {"mapping": {"a": 1}} if "mapping" in attributes else {"a": 1})
    spans = memory.get_finished_spans()
    provider.shutdown()

    class Noop(SpanCustomizer):
        def on_span_export(self, data):
            assert data["context"]["trace_state"] == {"k": "v"}
            return data

    exporter = OtelExporter(url=url, api_key="test")
    try:
        exporter.export(spans)
        set_span_customizers([Noop()])
        exporter.export(spans)
    finally:
        exporter.shutdown()
    assert "none_in_array" in spans[0].attributes
    baseline, customized = requests
    assert len(customized.resource_spans) == 1
    assert customized == baseline


@pytest.mark.parametrize("field", ["resource", "instrumentation_scope"])
def test_otel_customizer_preserves_encoder_valid_metadata(customized_otlp_receiver, caplog, field):
    from braintrust import SpanCustomizer, set_span_customizers
    from braintrust.otel import OtelExporter
    from opentelemetry.sdk.trace.export import SpanExportResult
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    scope_attributes_supported = hasattr(InstrumentationScope, "attributes")
    if field == "instrumentation_scope" and not scope_attributes_supported:
        pytest.skip("This OTel encoder does not export scope attributes")

    url, requests = customized_otlp_receiver
    spans = _sensitive_otel_spans()
    attributes = {"mixed": [1, "two"]}
    if scope_attributes_supported:
        attributes["nested"] = {"value": "safe"}

    class Customize(SpanCustomizer):
        def on_span_export(self, data):
            # Dictionary insertion order must not split equivalent OTLP groups.
            data[field]["attributes"] = dict(reversed(attributes.items())) if data["parent"] else attributes
            return data

    set_span_customizers([Customize()])
    exporter = OtelExporter(url=url, api_key="test")
    try:
        assert exporter.export(spans) == SpanExportResult.SUCCESS
    finally:
        exporter.shutdown()

    assert not caplog.records
    assert len(requests) == 1
    assert len(requests[0].resource_spans) == 1
    assert len(requests[0].resource_spans[0].scope_spans) == 1
    records = _wire_spans(requests)
    assert len(records) == 2
    for resource, scope, wire in records:
        assert wire.name in {"private.child", "private.parent"}
        metadata = resource.resource if field == "resource" else scope.scope
        attrs = {attribute.key: attribute.value for attribute in metadata.attributes}
        assert set(attrs) == set(attributes)
        mixed = attrs["mixed"].array_value.values
        assert len(mixed) == 2
        assert mixed[0].int_value == 1
        assert mixed[1].string_value == "two"
        if scope_attributes_supported:
            nested = attrs["nested"].kvlist_value.values
            assert len(nested) == 1
            assert nested[0].key == "value"
            assert nested[0].value.string_value == "safe"
    assert spans[0].resource.attributes["private.resource"] == "secret"


@pytest.mark.parametrize("field", ["resource", "instrumentation_scope"])
def test_otel_customizer_groups_equal_metadata_without_merging_distinct_values(customized_otlp_receiver, field):
    from braintrust import SpanCustomizer, set_span_customizers
    from braintrust.otel import OtelExporter
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor, SpanExportResult
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
    from opentelemetry.sdk.util.instrumentation import InstrumentationScope

    url, requests = customized_otlp_receiver
    variants = {
        "same-a": {"schema_url": "safe"},
        "same-b": {"schema_url": "safe"},
        "schema": {"schema_url": "different"},
    }
    if field == "resource" or hasattr(InstrumentationScope, "attributes"):
        variants.update(
            {
                "bool": {"schema_url": "safe", "attributes": {"value": True}},
                "int": {"schema_url": "safe", "attributes": {"value": 1}},
                "float": {"schema_url": "safe", "attributes": {"value": 1.0}},
                "array": {"schema_url": "safe", "attributes": {"value": [1]}},
            }
        )
    if field == "instrumentation_scope":
        variants.update({"name": {"name": "different"}, "version": {"version": "2"}})

    memory = InMemorySpanExporter()
    provider = TracerProvider(resource=Resource({}), shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("source", "1")
    for name in variants:
        tracer.start_span(name).end()
    provider.shutdown()

    class Customize(SpanCustomizer):
        def on_span_export(self, data):
            data[field] = variants[data["name"]]
            return data

    set_span_customizers([Customize()])
    exporter = OtelExporter(url=url, api_key="test")
    try:
        assert exporter.export(memory.get_finished_spans()) == SpanExportResult.SUCCESS
    finally:
        exporter.shutdown()

    assert len(requests) == 1
    if field == "resource":
        groups = [
            frozenset(span.name for scope in resource.scope_spans for span in scope.spans)
            for resource in requests[0].resource_spans
        ]
    else:
        assert len(requests[0].resource_spans) == 1
        groups = [frozenset(span.name for span in scope.spans) for scope in requests[0].resource_spans[0].scope_spans]
    assert len(groups) == len(variants) - 1
    assert set(groups) == {frozenset({"same-a", "same-b"})} | {
        frozenset({name}) for name in variants if name not in {"same-a", "same-b"}
    }


def test_otel_customizer_unexportable_span_strips_only_that_span(customized_otlp_receiver, caplog):
    import logging

    from braintrust import SpanCustomizer, set_span_customizers
    from braintrust.otel import OtelExporter, _SpanWithAttributes
    from opentelemetry.sdk.trace.export import SpanExportResult

    url, requests = customized_otlp_receiver
    healthy, original = _sensitive_otel_spans()
    broken = _SpanWithAttributes(original, {**original.attributes, "private": object()})
    calls = []

    class Observe(SpanCustomizer):
        def on_span_export(self, data):
            calls.append(data["name"])
            return data

    set_span_customizers([Observe()])
    exporter = OtelExporter(url=url, api_key="test")
    with caplog.at_level(logging.ERROR):
        try:
            assert exporter.export([healthy, broken]) == SpanExportResult.SUCCESS
        finally:
            exporter.shutdown()
    assert calls == ["private.child"]
    assert len(caplog.records) == 1 and "private" not in caplog.text
    wires = {wire.span_id: wire for _, _, wire in _wire_spans(requests)}
    assert wires[healthy.context.span_id.to_bytes(8, "big")].name == "private.child"
    stripped = wires[original.context.span_id.to_bytes(8, "big")]
    assert not stripped.name
    assert {a.key: a.value.string_value for a in stripped.attributes} == {
        "braintrust.context_json": '{"customizer_error":true}'
    }
