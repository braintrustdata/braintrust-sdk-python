"""Real Discovery Engine responses, recorded over REST and gRPC."""

import gc
import json
import os
import subprocess
import weakref
from contextlib import nullcontext
from pathlib import Path
from urllib.parse import urlsplit

import pytest
import yaml
from braintrust import auto_instrument, logger
from braintrust.conftest import get_vcr_config
from braintrust.test_helpers import init_test_logger


pytest.importorskip("google.cloud.discoveryengine_v1")

from google.auth.credentials import AnonymousCredentials
from google.cloud import discoveryengine_v1 as discoveryengine
from google.oauth2.credentials import Credentials


def _resource(request, cassette_dir, env_name, cassette_name, separator):
    if request.config.getoption("--vcr-record") == "all":
        value = os.getenv(f"BRAINTRUST_GOOGLE_DISCOVERYENGINE_{env_name}")
        if not value:
            pytest.fail(f"Set BRAINTRUST_GOOGLE_DISCOVERYENGINE_{env_name} to record Discovery Engine tests")
        return value
    cassette = yaml.safe_load((Path(cassette_dir) / cassette_name).read_text())
    resource = urlsplit(cassette["interactions"][0]["request"]["uri"]).path.removeprefix("/v1/")
    return resource.split(separator)[0]


@pytest.fixture
def LOCATION(request, vcr_cassette_dir):
    project = _resource(request, vcr_cassette_dir, "PROJECT", "test_rank.yaml", "/locations/")
    return (
        f"projects/{project}/locations/global"
        if request.config.getoption("--vcr-record") == "all"
        else f"{project}/locations/global"
    )


@pytest.fixture
def SERVING_CONFIG(request, vcr_cassette_dir, LOCATION):
    app = _resource(request, vcr_cassette_dir, "APP", "test_answer_query[False].yaml", ":answer")
    return (
        f"{LOCATION}/collections/default_collection/engines/{app}/servingConfigs/default_search"
        if request.config.getoption("--vcr-record") == "all"
        else app
    )


@pytest.fixture
def DATASTORE_CONFIG(request, vcr_cassette_dir, LOCATION):
    datastore = _resource(request, vcr_cassette_dir, "DATASTORE", "test_converse_conversation.yaml", "/conversations/")
    return (
        f"{LOCATION}/collections/default_collection/dataStores/{datastore}/servingConfigs/default_search"
        if request.config.getoption("--vcr-record") == "all"
        else f"{datastore}/servingConfigs/default_search"
    )


@pytest.fixture
def DATASTORE(DATASTORE_CONFIG):
    return DATASTORE_CONFIG.split("/dataStores/")[1].split("/")[0]


@pytest.fixture
def vcr_cassette_name(request):
    marker = request.node.get_closest_marker("vcr")
    return marker.args[0] if marker and marker.args else request.node.name


@pytest.fixture(scope="module")
def vcr_config():
    return {**get_vcr_config(), "match_on": ["method", "scheme", "host", "port", "path", "query", "body"]}


@pytest.fixture(scope="session")
def credentials(request):
    if request.config.getoption("--vcr-record") == "all":
        if not os.getenv("BRAINTRUST_GOOGLE_DISCOVERYENGINE_PROJECT"):
            pytest.fail("Set BRAINTRUST_GOOGLE_DISCOVERYENGINE_PROJECT to record Discovery Engine tests")
        # Refresh outside the recorded HTTP call; credentials never enter cassettes.
        token = subprocess.check_output(
            ["gcloud", "auth", "application-default", "print-access-token"], text=True
        ).strip()
        return Credentials(token=token)
    return AnonymousCredentials()


@pytest.fixture
def memory_logger():
    init_test_logger("test-discoveryengine")
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.fixture
def rank_request(LOCATION):
    return {
        "ranking_config": f"{LOCATION}/rankingConfigs/default_ranking_config",
        "model": "semantic-ranker-512@latest",
        "query": "What is Braintrust?",
        "records": [
            {"id": "1", "content": "Braintrust is a platform for evaluating and monitoring AI applications."},
            {"id": "2", "content": "The moon orbits the Earth."},
        ],
        "top_n": 1,
    }


@pytest.mark.vcr("test_rank.yaml")
@pytest.mark.parametrize("mode", ["manual", "manual_then_setup", "setup_then_manual"])
def test_rank(memory_logger, credentials, rank_request, mode):
    from braintrust.integrations.google_discoveryengine import (
        setup_google_discoveryengine,
        wrap_google_discoveryengine,
    )

    client = discoveryengine.RankServiceClient(transport="rest", credentials=credentials)
    untouched = discoveryengine.RankServiceClient(transport="rest", credentials=credentials)
    original = untouched.rank
    if mode == "setup_then_manual":
        assert setup_google_discoveryengine()
        assert setup_google_discoveryengine()
    assert wrap_google_discoveryengine(client) is client
    assert wrap_google_discoveryengine(client) is client
    if mode == "manual_then_setup":
        assert setup_google_discoveryengine()
    if mode == "manual":
        assert untouched.rank == original
        assert not hasattr(untouched.rank, "__wrapped__")
    result = client.rank(request=rank_request, retry=None)
    assert result.records[0].id == "1"
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "google_discoveryengine.rank"
    assert span["span_attributes"]["type"] == "task"
    assert span["metadata"]["provider"] == "google"
    assert span["metadata"]["model"] == "semantic-ranker-512@latest"
    assert span["input"]["query"] == "What is Braintrust?"
    assert span["output"][0]["id"] == "1"
    assert span["output"][0]["score"] == result.records[0].score
    assert not {"tokens", "prompt_tokens", "completion_tokens"} & span["metrics"].keys()
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "google-discoveryengine-auto"


QUERY = "What was Alphabet's revenue in 2022?"


def _assert_generation_span(memory_logger, method, text):
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == f"google_discoveryengine.{method}"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == "google"
    assert "model" not in span["metadata"]
    assert span["output"][0]["message"]["content"] == text
    assert not {"tokens", "prompt_tokens", "completion_tokens"} & span["metrics"].keys()
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "google-discoveryengine-auto"
    json.dumps({key: span[key] for key in ("input", "output", "metadata")})
    return span


@pytest.mark.vcr
@pytest.mark.parametrize("stream", [False, True])
def test_answer_query(memory_logger, credentials, stream, SERVING_CONFIG):
    auto_instrument()
    client = discoveryengine.ConversationalSearchServiceClient(transport="rest", credentials=credentials)
    request = discoveryengine.AnswerQueryRequest(
        serving_config=SERVING_CONFIG,
        query={"text": QUERY},
        answer_generation_spec={"include_citations": True},
    )
    if stream:
        chunks = list(client.stream_answer_query(request, retry=None, timeout=90))
        assert chunks[-1].answer.state == discoveryengine.Answer.State.SUCCEEDED
        text = chunks[-1].answer.answer_text
        method = "stream_answer_query"
    else:
        result = client.answer_query(request, retry=None, timeout=90)
        text = result.answer.answer_text
        method = "answer_query"
    assert text and "could not be generated" not in text
    span = _assert_generation_span(memory_logger, method, text)
    assert span["input"] == [{"role": "user", "content": QUERY}]
    assert span["output"][0]["citations"]
    assert span["output"][0]["references"]
    final_answer = chunks[-1].answer if stream else result.answer
    assert len(span["output"][0]["references"]) == len(final_answer.references)
    assert len(span["output"][0]["citations"]) == len(final_answer.citations)
    if stream:
        assert len([chunk for chunk in chunks if chunk.answer.answer_text]) > 1
        assert span["metrics"]["time_to_first_token"] >= 0


@pytest.mark.vcr
def test_converse_conversation(memory_logger, credentials, DATASTORE, DATASTORE_CONFIG, LOCATION):
    auto_instrument()
    client = discoveryengine.ConversationalSearchServiceClient(transport="rest", credentials=credentials)
    result = client.converse_conversation(
        request={
            "name": f"{LOCATION}/collections/default_collection/dataStores/{DATASTORE}/conversations/-",
            "query": {"input": QUERY},
            "serving_config": DATASTORE_CONFIG,
            "summary_spec": {"summary_result_count": 3, "include_citations": True},
        },
        retry=None,
        timeout=90,
    )
    assert result.reply.summary.summary_text
    span = _assert_generation_span(memory_logger, "converse_conversation", result.reply.summary.summary_text)
    choice = span["output"][0]
    assert "summary_with_metadata" not in choice
    assert len(choice["references"]) == len(result.reply.summary.summary_with_metadata.references)
    assert choice["citation_metadata"]


@pytest.mark.vcr
def test_check_grounding(memory_logger, credentials, LOCATION):
    auto_instrument()
    client = discoveryengine.GroundedGenerationServiceClient(transport="rest", credentials=credentials)
    result = client.check_grounding(
        request={
            "grounding_config": f"{LOCATION}/groundingConfigs/default_grounding_config",
            "answer_candidate": "Braintrust evaluates AI applications.",
            "facts": [{"fact_text": "Braintrust is a platform for evaluating AI applications."}],
        },
        retry=None,
        timeout=90,
    )
    assert result.support_score > 0
    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["span_attributes"]["name"] == "google_discoveryengine.check_grounding"
    assert spans[0]["span_attributes"]["type"] == "task"
    assert spans[0]["output"]["support_score"] == result.support_score


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "method",
    [
        "answer_query",
        "stream_answer_query",
        "converse_conversation",
        "check_grounding",
        "rank",
    ],
)
async def test_async_grpc(
    memory_logger,
    credentials,
    request,
    vcr_cassette_dir,
    method,
    DATASTORE,
    DATASTORE_CONFIG,
    LOCATION,
    SERVING_CONFIG,
):
    from braintrust.integrations.google_discoveryengine._test_grpc import grpc_cassette

    auto_instrument()
    if method in ("answer_query", "stream_answer_query", "converse_conversation"):
        client = discoveryengine.ConversationalSearchServiceAsyncClient(credentials=credentials)
        response_type = discoveryengine.AnswerQueryResponse
        payload = discoveryengine.AnswerQueryRequest(
            serving_config=SERVING_CONFIG, query={"text": QUERY}, answer_generation_spec={"include_citations": True}
        )
        if method == "converse_conversation":
            response_type = discoveryengine.ConverseConversationResponse
            payload = discoveryengine.ConverseConversationRequest(
                name=f"{LOCATION}/collections/default_collection/dataStores/{DATASTORE}/conversations/-",
                serving_config=DATASTORE_CONFIG,
                query={"input": QUERY},
                summary_spec={"summary_result_count": 3},
            )
    elif method == "rank":
        client = discoveryengine.RankServiceAsyncClient(credentials=credentials)
        response_type = discoveryengine.RankResponse
        payload = discoveryengine.RankRequest(
            ranking_config=f"{LOCATION}/rankingConfigs/default_ranking_config",
            query="What is Braintrust?",
            records=[{"id": "1", "content": "Braintrust evaluates AI applications."}],
        )
    else:
        client = discoveryengine.GroundedGenerationServiceAsyncClient(credentials=credentials)
        response_type = discoveryengine.CheckGroundingResponse
        payload = discoveryengine.CheckGroundingRequest(
            grounding_config=f"{LOCATION}/groundingConfigs/default_grounding_config",
            answer_candidate="Braintrust evaluates AI applications.",
            facts=[{"fact_text": "Braintrust is a platform for evaluating AI applications."}],
        )
    streaming = method.startswith("stream_")
    path = Path(vcr_cassette_dir) / f"test_async_grpc[{method}].json"

    try:
        with grpc_cassette(
            client,
            method,
            response_type,
            path,
            record=request.config.getoption("--vcr-record") == "all",
            streaming=streaming,
        ):
            result = await getattr(client, method)(payload, retry=None, timeout=90)
            if streaming:
                chunks = [chunk async for chunk in result]
                assert chunks
            else:
                assert isinstance(result, response_type)
        spans = memory_logger.pop()
        assert len(spans) == 1
        span = spans[0]
        assert span["span_attributes"]["name"] == f"google_discoveryengine.{method}"
        assert span["span_attributes"]["type"] == ("task" if method in ("rank", "check_grounding") else "llm")
        assert span["metadata"]["provider"] == "google"
        assert "model" not in span["metadata"]
        assert span["context"]["span_origin"]["instrumentation"]["name"] == "google-discoveryengine-auto"
        assert span["metrics"]["end"] >= span["metrics"]["start"]
        assert not {"tokens", "prompt_tokens", "completion_tokens"} & span["metrics"].keys()
        if method in ("answer_query", "stream_answer_query", "converse_conversation"):
            text = (
                chunks[-1].answer.answer_text
                if streaming
                else result.reply.summary.summary_text
                if method == "converse_conversation"
                else result.answer.answer_text
            )
            assert text
            assert span["input"] == [{"role": "user", "content": QUERY}]
            assert span["output"][0]["message"]["content"] == text
            if streaming:
                assert span["metrics"]["time_to_first_token"] >= 0
        elif method == "rank":
            assert span["output"][0]["id"] == result.records[0].id
            assert span["output"][0]["score"] == result.records[0].score
        elif method == "check_grounding":
            assert span["output"]["support_score"] == result.support_score
            assert span["input"]["answer_candidate"] == payload.answer_candidate
        json.dumps({key: span[key] for key in ("input", "output", "metadata")})
    finally:
        await client.transport.close()


@pytest.fixture(autouse=True)
def restore_methods():
    from braintrust.integrations.google_discoveryengine.patchers import PATCHERS

    originals = []
    for patcher in PATCHERS:
        class_name, method = patcher.target_path.split(".")
        cls = getattr(discoveryengine, class_name)
        original = getattr(cls, method)
        originals.append((cls, method, original, patcher.patch_marker_attr()))
    yield
    for cls, method, original, marker in originals:
        setattr(cls, method, original)
        if hasattr(original, marker):
            delattr(original, marker)


def test_patch_scope():
    import inspect

    from braintrust.integrations.google_discoveryengine import setup_google_discoveryengine
    from braintrust.integrations.google_discoveryengine.patchers import PATCHERS
    from google.cloud import discoveryengine_v1alpha, discoveryengine_v1beta

    untouched = [
        (discoveryengine.GroundedGenerationServiceClient, "generate_grounded_content"),
        (discoveryengine.GroundedGenerationServiceClient, "stream_generate_grounded_content"),
        (discoveryengine.GroundedGenerationServiceAsyncClient, "generate_grounded_content"),
        (discoveryengine.GroundedGenerationServiceAsyncClient, "stream_generate_grounded_content"),
        (discoveryengine.SearchServiceClient, "search"),
        (discoveryengine.SearchServiceClient, "search_lite"),
        (discoveryengine.AssistantServiceClient, "stream_assist"),
        (discoveryengine.ConversationalSearchServiceClient, "get_answer"),
        (discoveryengine.ConversationalSearchServiceClient, "create_conversation"),
        (discoveryengine_v1alpha.RankServiceClient, "rank"),
        (discoveryengine_v1beta.RankServiceClient, "rank"),
    ]
    originals = [inspect.getattr_static(cls, name) for cls, name in untouched]
    assert setup_google_discoveryengine()
    for (cls, name), original in zip(untouched, originals):
        assert inspect.getattr_static(cls, name) is original
    for patcher in PATCHERS:
        assert patcher.is_patched(discoveryengine, "0.20.3")
        target = patcher.resolve_target(discoveryengine, "0.20.3")
        assert hasattr(target, "__wrapped__")
        assert not hasattr(target.__wrapped__, "__wrapped__")


@pytest.mark.vcr("test_answer_query[True].yaml")
@pytest.mark.parametrize("consume", ["close", "abandon", "unstarted"])
def test_stream_lifecycle(memory_logger, credentials, SERVING_CONFIG, consume):
    from braintrust import current_span, start_span
    from braintrust.integrations.google_discoveryengine import wrap_google_discoveryengine

    client = wrap_google_discoveryengine(
        discoveryengine.ConversationalSearchServiceClient(transport="rest", credentials=credentials)
    )
    with start_span(name="caller") as parent:
        stream = client.stream_answer_query(
            discoveryengine.AnswerQueryRequest(
                serving_config=SERVING_CONFIG,
                query={"text": QUERY},
                answer_generation_spec={"include_citations": True},
            ),
            retry=None,
        )
        assert current_span() is parent
        chunks = []
        if consume != "unstarted":
            for chunk in stream:
                chunks.append(chunk)
                if chunk.answer.answer_text:
                    break
        assert current_span() is parent
        if consume == "close":
            stream.close()
            stream.close()
        stream_ref = weakref.ref(stream)
        del stream
        gc.collect()
        assert stream_ref() is None
        assert current_span() is parent
    spans = memory_logger.pop()
    assert len(spans) == 2
    child = next(
        span for span in spans if span["span_attributes"]["name"] == "google_discoveryengine.stream_answer_query"
    )
    parent_row = next(span for span in spans if span["span_attributes"]["name"] == "caller")
    assert child["span_parents"] == [parent_row["span_id"]]
    assert child["output"][0]["message"]["content"] == "".join(chunk.answer.answer_text for chunk in chunks)
    assert "end" in child["metrics"]
    gc.collect()
    assert memory_logger.pop() == []


def test_auto_instrument_subprocess():
    from braintrust.integrations.test_utils import verify_autoinstrument_script

    verify_autoinstrument_script("test_auto_google_discoveryengine.py")


@pytest.mark.vcr
@pytest.mark.parametrize("asynchronous_mode", [True, False])
def test_answer_requested_model(memory_logger, credentials, asynchronous_mode, SERVING_CONFIG):
    from braintrust.integrations.google_discoveryengine import setup_google_discoveryengine

    setup_google_discoveryengine()
    client = discoveryengine.ConversationalSearchServiceClient(transport="rest", credentials=credentials)
    from google.api_core.exceptions import BadRequest

    expected = (
        pytest.raises(BadRequest, match="asynchronous mode is deprecated") if asynchronous_mode else nullcontext()
    )
    with expected:
        result = client.answer_query(
            request={
                "serving_config": SERVING_CONFIG,
                "query": {"text": QUERY},
                "answer_generation_spec": {"model_spec": {"model_version": "stable"}},
                "asynchronous_mode": asynchronous_mode,
            },
            retry=None,
            timeout=90,
        )
    spans = memory_logger.pop()
    if asynchronous_mode:
        assert spans == []
    else:
        assert isinstance(result, discoveryengine.AnswerQueryResponse)
        assert len(spans) == 1
        assert spans[0]["metadata"]["model"] == "stable"


@pytest.mark.vcr("test_rank.yaml")
def test_normalization_failure_does_not_change_result(memory_logger, credentials, monkeypatch, rank_request):
    from braintrust.integrations.google_discoveryengine import setup_google_discoveryengine, tracing

    def broken(*args):
        raise ValueError("injected extraction failure")

    monkeypatch.setattr(tracing, "_prepare", broken)
    monkeypatch.setattr(tracing, "_output", broken)
    setup_google_discoveryengine()
    client = discoveryengine.RankServiceClient(transport="rest", credentials=credentials)
    result = client.rank(request=rank_request, retry=None)
    assert result.records[0].id == "1"
    span = memory_logger.pop()[0]
    assert "error" not in span


@pytest.mark.asyncio
@pytest.mark.parametrize("consume", ["read", "cancel", "aclose", "abandon", "unstarted"])
async def test_async_stream_lifecycle(memory_logger, credentials, vcr_cassette_dir, consume, SERVING_CONFIG):
    from braintrust import current_span, start_span
    from braintrust.integrations.google_discoveryengine import wrap_google_discoveryengine
    from braintrust.integrations.google_discoveryengine._test_grpc import grpc_cassette
    from grpc.aio import EOF

    client = wrap_google_discoveryengine(
        discoveryengine.ConversationalSearchServiceAsyncClient(credentials=credentials)
    )
    payload = discoveryengine.AnswerQueryRequest(
        serving_config=SERVING_CONFIG,
        query={"text": QUERY},
        answer_generation_spec={"include_citations": True},
    )
    path = Path(vcr_cassette_dir) / "test_async_grpc[stream_answer_query].json"
    try:
        with (
            start_span(name="caller") as parent,
            grpc_cassette(
                client,
                "stream_answer_query",
                discoveryengine.AnswerQueryResponse,
                path,
                streaming=True,
            ),
        ):
            stream = await client.stream_answer_query(payload, retry=None)
            assert current_span() is parent
            chunks = []
            if consume == "read":
                while (chunk := await stream.read()) is not EOF:
                    chunks.append(chunk)
                    assert current_span() is parent
                assert await stream.read() is EOF
            elif consume != "unstarted":
                async for chunk in stream:
                    chunks.append(chunk)
                    if chunk.answer.answer_text:
                        break
                if consume == "cancel":
                    assert stream.cancel()
                elif consume == "aclose":
                    await stream.aclose()
                if consume in ("cancel", "aclose"):
                    assert stream.cancelled()
            stream_ref = weakref.ref(stream)
            del stream
            gc.collect()
            assert stream_ref() is None
            assert current_span() is parent
        spans = memory_logger.pop()
        assert len(spans) == 2
        child = next(
            span for span in spans if span["span_attributes"]["name"] == "google_discoveryengine.stream_answer_query"
        )
        parent_row = next(span for span in spans if span["span_attributes"]["name"] == "caller")
        assert child["span_parents"] == [parent_row["span_id"]]
        expected_text = (
            chunks[-1].answer.answer_text
            if consume == "read"
            else "".join(chunk.answer.answer_text for chunk in chunks)
        )
        assert child["output"][0]["message"]["content"] == expected_text
        assert "end" in child["metrics"]
        gc.collect()
        assert memory_logger.pop() == []
    finally:
        await client.transport.close()


@pytest.mark.vcr
def test_rank_output_limit(memory_logger, credentials, LOCATION):
    from braintrust.integrations.google_discoveryengine import setup_google_discoveryengine

    setup_google_discoveryengine()
    client = discoveryengine.RankServiceClient(transport="rest", credentials=credentials)
    result = client.rank(
        request={
            "ranking_config": f"{LOCATION}/rankingConfigs/default_ranking_config",
            "query": "AI evaluation",
            "records": [{"id": str(i), "content": f"Evaluation example {i}."} for i in range(101)],
        },
        retry=None,
    )
    assert len(result.records) == 101
    span = memory_logger.pop()[0]
    assert len(span["input"]["records"]) == 101
    assert len(span["output"]) == 100
    assert [record["id"] for record in span["output"]] == [result.records[i].id for i in range(100)]


@pytest.fixture
def error_request(stream, LOCATION, SERVING_CONFIG):
    if stream:
        return {"serving_config": SERVING_CONFIG}
    return {
        "ranking_config": f"{LOCATION}/rankingConfigs/default_ranking_config",
        "query": "test",
        "records": [{"id": "empty"}],
    }


@pytest.mark.vcr
@pytest.mark.parametrize(
    "stream,vcr_cassette_name",
    [(False, "test_provider_error"), (True, "test_stream_provider_error")],
)
def test_provider_error(memory_logger, credentials, stream, error_request, vcr_cassette_name):
    from braintrust.integrations.google_discoveryengine import setup_google_discoveryengine
    from google.api_core.exceptions import BadRequest, InternalServerError

    setup_google_discoveryengine()
    client_type = discoveryengine.ConversationalSearchServiceClient if stream else discoveryengine.RankServiceClient
    client = client_type(transport="rest", credentials=credentials)
    method = client.stream_answer_query if stream else client.rank
    with pytest.raises(InternalServerError if stream else BadRequest):
        result = method(request=error_request, retry=None)
        if stream:
            list(result)
    spans = memory_logger.pop()
    assert len(spans) == 1
    assert spans[0]["error"]
    assert spans[0]["metrics"]["end"] >= spans[0]["metrics"]["start"]


@pytest.mark.asyncio
@pytest.mark.parametrize("stream", [False, True])
async def test_async_provider_error(memory_logger, credentials, request, vcr_cassette_dir, stream, error_request):
    from braintrust.integrations.google_discoveryengine import setup_google_discoveryengine
    from braintrust.integrations.google_discoveryengine._test_grpc import grpc_cassette
    from google.api_core.exceptions import InternalServerError, InvalidArgument

    setup_google_discoveryengine()
    client_type = (
        discoveryengine.ConversationalSearchServiceAsyncClient if stream else discoveryengine.RankServiceAsyncClient
    )
    client = client_type(credentials=credentials)
    method = "stream_answer_query" if stream else "rank"
    cassette = "test_async_stream_provider_error.json" if stream else "test_async_provider_error.json"
    try:
        with grpc_cassette(
            client,
            method,
            discoveryengine.AnswerQueryResponse if stream else discoveryengine.RankResponse,
            Path(vcr_cassette_dir) / cassette,
            record=request.config.getoption("--vcr-record") == "all",
            streaming=stream,
        ):
            with pytest.raises(InternalServerError if stream else InvalidArgument):
                result = await getattr(client, method)(request=error_request, retry=None)
                if stream:
                    async for _ in result:
                        pass
        spans = memory_logger.pop()
        assert len(spans) == 1
        assert spans[0]["error"]
        assert spans[0]["metrics"]["end"] >= spans[0]["metrics"]["start"]
    finally:
        await client.transport.close()
