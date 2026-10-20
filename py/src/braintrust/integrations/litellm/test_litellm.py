import asyncio
import base64
import os
import re
import time
import uuid
from pathlib import Path
from types import SimpleNamespace

import litellm
import pytest
from braintrust import Attachment, logger
from braintrust.conftest import get_vcr_config
from braintrust.integrations.litellm import patch_litellm, wrap_litellm
from braintrust.integrations.test_decisions import (
    DECISIONS_INPUT,
    DECISIONS_MODEL,
    DECISIONS_QUESTIONS,
    assert_decisions_image_span,
    assert_decisions_span,
    decisions_image_input,
)
from braintrust.integrations.test_utils import (
    assert_metrics_are_valid,
    verify_autoinstrument_script,
)
from braintrust.integrations.versioning import detect_module_version, version_satisfies
from braintrust.test_helpers import assert_dict_matches, init_test_logger


TEST_ORG_ID = "test-org-litellm-py-tracing"
PROJECT_NAME = "test-project-litellm-py-tracing"
TEST_MODEL = "gpt-4o-mini"  # cheapest model for tests
TEST_TEXT_MODEL = "anthropic/claude-haiku-4-5-20251001"
TEST_PROMPT = "What's 12 + 12?"
TEST_CACHE_MODEL = "anthropic/claude-haiku-4-5-20251001"
TEST_CACHEABLE_PROMPT = "Braintrust LiteLLM prompt caching regression context. " * 400
TEST_AUDIO_FILE = os.path.join(os.path.dirname(__file__), "..", "..", "fixtures", "test_audio.wav")


@pytest.fixture(scope="module")
def vcr_config():
    # The LiteLLM 1.103.0 win_amd64 wheel ships its vendored tiktoken rank files
    # with CRLF line endings, so tiktoken's hash check fails and it re-downloads
    # cl100k_base on first use. Let that fetch through rather than record a
    # ~1.7MB blob into every cassette.
    config = get_vcr_config()
    config["ignore_hosts"] = ["openaipublic.blob.core.windows.net"]

    def scrub_cloudflare_account(request):
        if request.host == "api.cloudflare.com":
            request.uri = re.sub(r"/accounts/[^/]+/", "/accounts/test-account/", request.uri)
        return request

    config["before_record_request"] = scrub_cloudflare_account
    return config


RERANK_MODEL = "cohere/rerank-english-v3.0"
RERANK_QUERY = "What is the capital of France?"
RERANK_DOCUMENTS = [
    "Paris is the capital of France.",
    "Berlin is the capital of Germany.",
    "Madrid is the capital of Spain.",
]

# Applied to every sync/async pair below. The ``is_async`` param selects the
# async LiteLLM entrypoint (e.g. ``acompletion``) over the sync one; the two
# variants share a single test body and record to ``...[sync]`` / ``...[async]``
# cassettes.
#
# The parametrized tests are plain (non-``async``) functions so the ``sync``
# variant calls LiteLLM with no running event loop, exactly as ordinary
# synchronous callers do -- ``_run``/``_run_stream`` drive the ``async`` variant
# with ``asyncio.run`` instead. Running the sync entrypoints inside a coroutine
# would let LiteLLM take loop-dependent code paths and hide sync-only failures.
sync_async = pytest.mark.parametrize("is_async", [False, True], ids=["sync", "async"])


def _run(is_async, sync_fn, async_fn, *args, **kwargs):
    """Call ``sync_fn`` directly, or drive ``async_fn`` with ``asyncio.run``."""
    if is_async:
        return asyncio.run(async_fn(*args, **kwargs))
    return sync_fn(*args, **kwargs)


def _run_stream(is_async, sync_fn, async_fn, *args, **kwargs):
    """Open a streaming request and drain it into a list of chunks."""
    if is_async:

        async def _drain():
            stream = await async_fn(*args, **kwargs)
            return [chunk async for chunk in stream]

        return asyncio.run(_drain())
    return list(sync_fn(*args, **kwargs))


def _run_stream_mutating_chunks(is_async, sync_fn, async_fn, *args, **kwargs):
    """Drain a stream while mutating each yielded content delta after capture."""
    if is_async:

        async def _drain():
            stream = await async_fn(*args, **kwargs)
            chunks = []
            content = []
            async for chunk in stream:
                chunks.append(chunk)
                if chunk.choices and chunk.choices[0].delta.content:
                    content.append(chunk.choices[0].delta.content)
                    chunk.choices[0].delta.content = "mutated by consumer"
            return chunks, "".join(content)

        return asyncio.run(_drain())

    chunks = []
    content = []
    for chunk in sync_fn(*args, **kwargs):
        chunks.append(chunk)
        if chunk.choices and chunk.choices[0].delta.content:
            content.append(chunk.choices[0].delta.content)
            chunk.choices[0].delta.content = "mutated by consumer"
    return chunks, "".join(content)


def _assert_speech_output_attachment(span) -> None:
    assert span["output"]["type"] == "audio"
    assert span["output"]["audio_size_bytes"] > 0
    attachment = span["output"]["file"]["file_data"]
    assert isinstance(attachment, Attachment)
    assert attachment.reference["content_type"].startswith("audio/")
    assert attachment.reference["filename"].startswith("generated_speech")


@pytest.fixture(autouse=True)
def _patch():
    patch_litellm()


@pytest.fixture
def memory_logger():
    init_test_logger(PROJECT_NAME)
    with logger._internal_with_memory_background_logger() as bgl:
        yield bgl


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
@sync_async
@pytest.mark.parametrize("stream", [False, True], ids=["blocking", "streaming"])
def test_litellm_image_input_materializes_attachment(memory_logger, is_async, stream):
    """Completion spans capture output, usage, and attachments in every call mode."""
    image_base64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8DwHwAFBQIAX8jx0gAAAABJRU5ErkJggg=="
    image_url = f"data:image/png;base64,{image_base64}"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "What color is this image?"},
                {"type": "image_url", "image_url": {"url": image_url, "detail": "low"}},
            ],
        }
    ]
    run = _run_stream_mutating_chunks if stream else _run
    start = time.time()
    response = run(
        is_async, litellm.completion, litellm.acompletion, model=TEST_MODEL, messages=messages, stream=stream
    )
    end = time.time()
    if stream:
        response, content = response
        assert len(response) > 1
    else:
        content = response.choices[0].message.content
    assert content
    assert messages[0]["content"][1]["image_url"] == {"url": image_url, "detail": "low"}
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "litellm-auto"
    assert span["span_attributes"]["name"] == "Completion"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["model"] == TEST_MODEL
    assert span["metadata"]["provider"] == "openai"
    assert_metrics_are_valid(span["metrics"], start, end)
    assert span["metrics"]["time_to_first_token"] >= 0
    assert span["output"][0]["message"]["content"] == content
    assert span["input"][0]["role"] == "user"
    assert span["input"][0]["content"][0] == messages[0]["content"][0]
    logged_image = span["input"][0]["content"][1]["image_url"]
    assert logged_image["detail"] == "low"
    attachment = logged_image["url"]
    assert isinstance(attachment, Attachment)
    assert attachment.data == base64.b64decode(image_base64)
    assert attachment.reference["content_type"] == "image/png"


@pytest.mark.vcr
def test_litellm_prompt_caching_metrics(memory_logger) -> None:
    package_version = os.environ.get("BRAINTRUST_TEST_PACKAGE_VERSION", "direct")
    cacheable_prompt = f"{TEST_CACHEABLE_PROMPT}\n\nCache salt: {uuid.uuid4().hex}\n"
    messages = [
        {
            "role": "system",
            "content": [
                {
                    "type": "text",
                    "text": cacheable_prompt,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
        },
        {"role": "user", "content": "Reply with OK."},
    ]

    response = litellm.completion(model=TEST_CACHE_MODEL, messages=messages, max_tokens=5)
    assert response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    first_metrics = spans[0]["metrics"]
    first_has_ttl_breakdown = (
        "prompt_cache_creation_5m_tokens" in first_metrics or "prompt_cache_creation_1h_tokens" in first_metrics
    )
    first_cache_creation_tokens = (
        first_metrics.get("prompt_cache_creation_5m_tokens", 0)
        + first_metrics.get("prompt_cache_creation_1h_tokens", 0)
        if first_has_ttl_breakdown
        else first_metrics.get("prompt_cache_creation_tokens", 0)
    )

    assert first_cache_creation_tokens > 0
    assert first_metrics["prompt_cached_tokens"] == 0
    if first_has_ttl_breakdown:
        assert "prompt_cache_creation_tokens" not in first_metrics

    second_metrics = None
    for attempt in range(3):
        response = litellm.completion(model=TEST_CACHE_MODEL, messages=messages, max_tokens=5)
        assert response.choices[0].message.content

        spans = memory_logger.pop()
        assert len(spans) == 1
        second_metrics = spans[0]["metrics"]
        if second_metrics.get("prompt_cached_tokens", 0) > 0:
            break
        if attempt < 2:
            time.sleep(1)

    assert second_metrics is not None
    assert second_metrics["prompt_cached_tokens"] > 0
    second_has_ttl_breakdown = (
        "prompt_cache_creation_5m_tokens" in second_metrics or "prompt_cache_creation_1h_tokens" in second_metrics
    )
    if second_has_ttl_breakdown:
        assert "prompt_cache_creation_tokens" not in second_metrics

    if package_version == "latest":
        assert first_has_ttl_breakdown
        assert second_has_ttl_breakdown

    unsupported_cache_metrics = {
        "cache_creation_input_tokens",
        "cache_read_input_tokens",
        "prompt_cache_write_tokens",
    }
    assert unsupported_cache_metrics.isdisjoint(first_metrics)
    assert unsupported_cache_metrics.isdisjoint(second_metrics)


@pytest.mark.vcr
@sync_async
def test_litellm_text_completion_metrics(memory_logger, is_async) -> None:
    assert not memory_logger.pop()

    start = time.time()
    response = _run(
        is_async,
        litellm.text_completion,
        litellm.atext_completion,
        model=TEST_TEXT_MODEL,
        prompt=TEST_PROMPT,
    )
    end = time.time()

    assert response
    assert response.choices[0].text
    assert "24" in response.choices[0].text or "twenty-four" in response.choices[0].text.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["model"] == TEST_TEXT_MODEL
    assert span["metadata"]["provider"] == "anthropic"
    assert TEST_PROMPT in str(span["input"])
    assert "text" in span["output"][0]


@pytest.mark.vcr
@sync_async
def test_litellm_decisions_metrics(memory_logger, is_async):
    if not version_satisfies(detect_module_version(litellm, ("litellm",)), ">=1.105.0"):
        pytest.skip("Decisions requires LiteLLM >=1.105.0")

    model = f"openai/{DECISIONS_MODEL}"
    start = time.time()
    response = _run(
        is_async,
        litellm.decisions,
        litellm.adecisions,
        model,
        DECISIONS_INPUT,
        DECISIONS_QUESTIONS,
        safety_identifier="do-not-log-this-identifier",
    )
    end = time.time()

    assert_decisions_span(
        memory_logger.pop(),
        response,
        name="Decision",
        model=model,
        instrumentation="litellm-auto",
        start=start,
        end=end,
    )


@pytest.mark.vcr(match_on=["method", "scheme", "host", "port", "path", "query", "body"])
@sync_async
def test_litellm_decisions_image(memory_logger, is_async):
    if not version_satisfies(detect_module_version(litellm, ("litellm",)), ">=1.105.0"):
        pytest.skip("Decisions requires LiteLLM >=1.105.0")

    messages = decisions_image_input()
    response = _run(
        is_async,
        litellm.decisions,
        litellm.adecisions,
        model=f"openai/{DECISIONS_MODEL}",
        input=messages,
        questions=DECISIONS_QUESTIONS[:1],
    )
    assert_decisions_image_span(memory_logger.pop(), response, messages)


def _assert_decision_media_span(memory_logger, response, model):
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert_dict_matches(span["span_attributes"], {"name": "Decision", "type": "decision"})
    assert span["metadata"] == {"model": model, "provider": model.split("/", 1)[0]}
    assert span["context"]["span_origin"]["instrumentation"]["name"] == "litellm-auto"
    assert span["metrics"]["start"] <= span["metrics"]["end"]
    assert all(span["metrics"][key] >= 0 for key in ("tokens", "prompt_tokens", "completion_tokens"))
    assert span["metrics"]["tokens"] == response.usage.input_tokens + response.usage.output_tokens
    assert span["metrics"]["prompt_tokens"] == response.usage.input_tokens
    assert span["metrics"]["completion_tokens"] == response.usage.output_tokens
    return span


@pytest.mark.vcr(
    cassette_name="test_litellm_decisions_media[decisions-sync]",
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)
@pytest.mark.parametrize(
    "is_async,systemone,conversion_fails",
    [
        # Exercise each API and call mode; conversion fallback is shared.
        pytest.param(False, False, False, id="decisions-sync"),
        pytest.param(True, True, False, id="systemone-async"),
        pytest.param(False, False, True, id="conversion-failure"),
    ],
)
def test_litellm_decisions_media(memory_logger, monkeypatch, is_async, systemone, conversion_fails):
    if not version_satisfies(detect_module_version(litellm, ("litellm",)), ">=1.106.0"):
        pytest.skip("Decision audio and video require LiteLLM >=1.106.0")

    from litellm.types.decisions import DecisionsBase64Video

    if conversion_fails:
        from braintrust.integrations.litellm import tracing

        materialize = tracing._materialize_attachment

        def fail_video(value, **kwargs):
            if kwargs.get("mime_type") == "video/mp4":
                raise RuntimeError("attachment conversion failed")
            return materialize(value, **kwargs)

        monkeypatch.setattr(tracing, "_materialize_attachment", fail_video)

    audio_bytes = Path(TEST_AUDIO_FILE).read_bytes()
    video_bytes = (Path(TEST_AUDIO_FILE).parent / "decision-video.mp4").read_bytes()
    audio_url = "data:audio/wav;base64," + base64.b64encode(audio_bytes).decode()
    audio = [audio_url]
    videos = [DecisionsBase64Video(content_type="video/mp4", base64=base64.b64encode(video_bytes).decode())]
    model = "cloudflare/clef-omni"
    instructions = "Does the video show a red background?"
    questions = (
        {"red": {"type": "noul", "instructions": instructions}}
        if systemone
        else [{"type": "predicate", "name": "red", "instructions": instructions}]
    )
    # Use the real account only when recording; playback needs no credentials.
    account_id = os.getenv("CLOUDFLARE_ACCOUNT_ID", "test-account")
    response = _run(
        is_async,
        litellm.systemone if systemone else litellm.decisions,
        litellm.asystemone if systemone else litellm.adecisions,
        model,
        "Inspect the supplied audio and video.",
        questions,
        audio=audio,
        videos=videos,
        api_base=f"https://api.cloudflare.com/client/v4/accounts/{account_id}/ai/run",
        api_key=os.getenv("CLOUDFLARE_API_TOKEN", "test-cloudflare-key"),
    )
    span = _assert_decision_media_span(memory_logger, response, model)
    assert set(span["input"]) == {"state", "questions"}
    state = span["input"]["state"]
    assert state["state"] == "Inspect the supplied audio and video."
    if conversion_fails:
        assert state == {
            "state": "Inspect the supplied audio and video.",
            "audio": audio,
            "videos": [video.model_dump() for video in videos],
        }
    else:
        for field, data, content_type in (("audio", audio_bytes, "audio/wav"), ("videos", video_bytes, "video/mp4")):
            attachment = state[field][0]
            assert isinstance(attachment, Attachment)
            assert attachment.data == data
            assert attachment.reference["content_type"] == content_type
    kind = "noul" if systemone else "boolean"
    answer = response.answers["red"].noul if systemone else response.answers[0].probability
    assert span["input"]["questions"] == [{"id": "red", "type": kind, "instructions": instructions}]
    assert span["output"] == {"answers": [{"id": "red", "type": kind, "noul" if systemone else "probability": answer}]}
    assert audio == [audio_url]
    assert base64.b64decode(videos[0].base64) == video_bytes


@pytest.mark.parametrize(
    "media_kind,systemone,is_async",
    [
        pytest.param(
            media_kind,
            systemone,
            is_async,
            id=f"{media_kind}-{'systemone' if systemone else 'decisions'}-{'async' if is_async else 'sync'}",
            marks=pytest.mark.vcr(
                cassette_name=f"test_litellm_decisions_state_media[{cassette_case}]",
                match_on=["method", "scheme", "host", "port", "path", "query", "body"],
            ),
        )
        for media_kind, systemone, is_async, cassette_case in (
            ("images", False, False, "images-decisions-sync"),
            ("assets", False, True, "assets-decisions-async"),
            ("images", True, True, "images-decisions-sync"),
            ("assets", True, False, "assets-decisions-async"),
        )
    ],
)
def test_litellm_decisions_state_media(memory_logger, is_async, systemone, media_kind):
    if not version_satisfies(detect_module_version(litellm, ("litellm",)), ">=1.106.0"):
        pytest.skip("Decisions state media forwarding requires LiteLLM >=1.106.0")

    from litellm.types.decisions import DecisionsAsset, DecisionsBase64Image

    image_bytes = (Path(TEST_AUDIO_FILE).parent / "decision-box.png").read_bytes()
    image_base64 = base64.b64encode(image_bytes).decode()
    image_url = "data:image/png;base64," + image_base64
    state = {"description": "Inspect this image: photo"}
    instructions = "Does the image show a red object?"
    questions = {"red": {"type": "noul", "instructions": instructions}}
    if media_kind == "images":
        model = f"openai/{DECISIONS_MODEL}"
        media = {"images": [image_url, DecisionsBase64Image(content_type="image/png", base64=image_base64)]}
        extra_headers = {}
    else:
        model = "typesafe/jev-1.14.0"
        media = {"assets": {"photo": DecisionsAsset(type="image", format="png", data_base64=image_base64)}}
        extra_headers = {"TypeSafe-Danger-Zone-Images": "true"}
    response = _run(
        is_async,
        litellm.systemone if systemone else litellm.decisions,
        litellm.asystemone if systemone else litellm.adecisions,
        model=model,
        state=state,
        questions=questions,
        extra_headers=extra_headers,
        **media,
    )
    span = _assert_decision_media_span(memory_logger, response, model)
    assert set(span["input"]) == {"state", "questions"}
    assert span["input"]["state"]["state"] == state
    captured = span["input"]["state"][media_kind]
    attachments = captured if media_kind == "images" else captured.values()
    for attachment in attachments:
        assert isinstance(attachment, Attachment)
        assert attachment.data == image_bytes
        assert attachment.reference["content_type"] == "image/png"
    assert span["input"]["questions"] == [{"id": "red", "type": "noul", "instructions": instructions}]
    assert span["output"] == {"answers": [{"id": "red", "type": "noul", "noul": response.answers["red"].noul}]}
    assert state == {"description": "Inspect this image: photo"}
    assert questions == {"red": {"type": "noul", "instructions": instructions}}
    if media_kind == "images":
        assert media["images"][0] == image_url
        assert media["images"][1].base64 == image_base64
    else:
        assert media["assets"]["photo"].data_base64 == image_base64


@pytest.mark.vcr(
    cassette_name="test_litellm_systemone_metrics[async]",
    match_on=["method", "scheme", "host", "port", "path", "query", "body"],
)
@sync_async
def test_litellm_systemone_metrics(memory_logger, is_async):
    if not version_satisfies(detect_module_version(litellm, ("litellm",)), ">=1.105.0"):
        pytest.skip("System One requires LiteLLM >=1.105.0")

    model = f"openai/{DECISIONS_MODEL}"
    questions = {"positive": {"type": "noul", "instructions": "Is the sentiment positive?"}}
    # Wrap the real provider entrypoints without the package-level patch.
    # Repeating setup must still produce exactly one span.
    module = SimpleNamespace(systemone=litellm.systemone.__wrapped__, asystemone=litellm.asystemone.__wrapped__)
    wrap_litellm(module)
    wrap_litellm(module)
    response = _run(is_async, module.systemone, module.asystemone, model, DECISIONS_INPUT, questions)
    span = _assert_decision_media_span(memory_logger, response, model)
    assert span["input"] == {
        "state": DECISIONS_INPUT,
        "questions": [{"id": "positive", **questions["positive"]}],
    }
    assert span["output"] == {
        "answers": [{"id": "positive", "type": "noul", "noul": response.answers["positive"].noul}]
    }


@pytest.mark.parametrize(
    "stream,is_async",
    [
        pytest.param(
            stream,
            is_async,
            id=f"{'streaming' if stream else 'blocking'}-{'async' if is_async else 'sync'}",
            marks=pytest.mark.vcr(
                cassette_name=f"test_litellm_responses_{'streaming' if stream else 'metrics'}[{'async' if is_async else 'sync'}]"
            ),
        )
        for stream in (False, True)
        for is_async in (False, True)
    ],
)
def test_litellm_responses(memory_logger, is_async, stream):
    """Reuse the existing blocking/streaming recordings for the Responses API."""
    run = _run_stream if stream else _run
    params = {"stream": True} if stream else {"instructions": "Just the number please"}
    start = time.time()
    response = run(
        is_async,
        litellm.responses,
        litellm.aresponses,
        model=TEST_MODEL,
        input=TEST_PROMPT,
        **params,
    )
    end = time.time()

    if stream:
        assert len(response) > 1
        content = "".join(
            chunk.delta for chunk in response if getattr(chunk, "type", None) == "response.output_text.delta"
        )
    else:
        content = response.output[0].content[0].text
    assert "24" in content or "twenty-four" in content.lower()

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    metrics = span["metrics"]
    assert_metrics_are_valid(metrics, start, end)
    assert span["metadata"]["model"] == TEST_MODEL
    assert span["metadata"]["provider"] == "openai"
    assert TEST_PROMPT in str(span["input"])
    assert "24" in str(span["output"]) or "twenty-four" in str(span["output"]).lower()
    if stream:
        assert span["metadata"]["stream"] is True


@pytest.mark.vcr
@sync_async
def test_litellm_embedding(memory_logger, is_async):
    assert not memory_logger.pop()

    response = _run(
        is_async,
        litellm.embedding,
        litellm.aembedding,
        model="text-embedding-ada-002",
        input="This is a test",
    )

    assert response
    assert response.data
    assert response.data[0]["embedding"]

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    assert span["metadata"]["model"] == "text-embedding-ada-002"
    assert span["metadata"]["provider"] == "openai"
    assert "This is a test" in str(span["input"])


@pytest.mark.vcr
@sync_async
def test_litellm_moderation(memory_logger, is_async):
    assert not memory_logger.pop()

    response = _run(
        is_async,
        litellm.moderation,
        litellm.amoderation,
        model="omni-moderation-latest",
        input="This is a test message",
    )

    assert response
    assert response.results

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span
    assert span["metadata"]["model"] == "omni-moderation-latest"
    assert span["metadata"]["provider"] == "openai"
    assert "This is a test message" in str(span["input"])


@pytest.mark.vcr
@sync_async
def test_litellm_image_generation(memory_logger, is_async):
    assert not memory_logger.pop()

    # Distinct prompts per variant to match the recorded [sync]/[async] cassettes.
    prompt = "A tiny blue square on a white background" if is_async else "A tiny red square on a white background"

    response = _run(
        is_async,
        litellm.image_generation,
        litellm.aimage_generation,
        model="gpt-image-1-mini",
        prompt=prompt,
        size="1024x1024",
    )

    assert response
    assert response.data
    assert response.data[0].b64_json or response.data[0].url

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "gpt-image-1-mini"
    assert span["metadata"]["provider"] == "openai"
    assert span["input"] == prompt
    assert span["output"]["images_count"] == 1
    assert span["metrics"]["duration"] >= 0


@pytest.mark.vcr
@sync_async
def test_litellm_transcription(memory_logger, is_async):
    assert not memory_logger.pop()

    with open(TEST_AUDIO_FILE, "rb") as f:
        response = _run(is_async, litellm.transcription, litellm.atranscription, model="whisper-1", file=f)

    assert response
    assert response.text == "you"

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "whisper-1"
    assert span["metadata"]["provider"] == "openai"
    assert isinstance(span["input"]["file"], Attachment)
    assert span["input"]["file"].reference["filename"] == "test_audio.wav"
    assert span["input"]["file"].reference["content_type"] in ("audio/x-wav", "audio/wav")  # OS-dependent
    assert span["output"] == "you"


@pytest.mark.vcr
@sync_async
def test_litellm_speech(memory_logger, is_async):
    assert not memory_logger.pop()

    response = _run(
        is_async,
        litellm.speech,
        litellm.aspeech,
        model="tts-1",
        voice="alloy",
        input="Hello, this is a test.",
        response_format="mp3",
    )

    assert response
    assert response.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["metadata"]["model"] == "tts-1"
    assert span["metadata"]["voice"] == "alloy"
    assert span["metadata"]["response_format"] == "mp3"
    assert span["metadata"]["provider"] == "openai"
    assert span["input"] == "Hello, this is a test."
    _assert_speech_output_attachment(span)


@sync_async
def test_litellm_completion_error(memory_logger, is_async):
    assert not memory_logger.pop()

    # Use a non-existent model to force an error
    fake_model = "non-existent-model"

    with pytest.raises(litellm.BadRequestError):
        _run(
            is_async,
            litellm.completion,
            litellm.acompletion,
            model=fake_model,
            messages=[{"role": "user", "content": TEST_PROMPT}],
        )

    logs = memory_logger.pop()
    assert len(logs) == 1
    log = logs[0]
    assert log["project_id"] == PROJECT_NAME
    assert log["metadata"]["model"] == fake_model


@pytest.mark.vcr
@pytest.mark.asyncio
async def test_litellm_async_parallel_requests(memory_logger):
    """Test multiple parallel async requests."""
    assert not memory_logger.pop()

    # Create multiple prompts
    prompts = [f"What is {i} + {i}?" for i in range(3, 6)]

    # Run requests in parallel
    tasks = [
        litellm.acompletion(model=TEST_MODEL, messages=[{"role": "user", "content": prompt}]) for prompt in prompts
    ]

    # Wait for all to complete
    results = await asyncio.gather(*tasks)

    # Check all results
    assert len(results) == 3
    for result in results:
        assert result.choices[0].message.content

    # Check that all spans were created
    spans = memory_logger.pop()
    assert len(spans) == 3

    # Verify each span has proper data
    for i, span in enumerate(spans):
        assert span["metadata"]["model"] == TEST_MODEL
        assert span["metadata"]["provider"] == "openai"
        assert prompts[i] in str(span["input"])
        assert_metrics_are_valid(span["metrics"])


@pytest.mark.vcr
def test_litellm_tool_calls(memory_logger):
    """Test tool calls with LiteLLM."""
    assert not memory_logger.pop()

    # Define tools that can be called
    tools = [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the weather for a location",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string", "description": "The location to get weather for"}},
                    "required": ["location"],
                },
            },
        },
    ]

    start = time.time()
    response = litellm.completion(
        model=TEST_MODEL,
        messages=[{"role": "user", "content": "What's the weather in New York?"}],
        tools=tools,
        temperature=0,
    )
    end = time.time()

    assert response
    assert response.choices

    # Verify spans were created
    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]

    # Validate the span structure
    assert_dict_matches(
        span,
        {
            "span_attributes": {"type": "llm", "name": "Completion"},
            "metadata": {
                "model": TEST_MODEL,
                "provider": "openai",
                "tools": lambda tools_list: (
                    len(tools_list) == 1
                    and any(tool.get("function", {}).get("name") == "get_weather" for tool in tools_list)
                ),
            },
            "input": lambda inp: "What's the weather in New York?" in str(inp),
            "metrics": lambda m: assert_metrics_are_valid(m, start, end) is None,
        },
    )


@pytest.mark.vcr
def test_litellm_openrouter_no_booleans_in_metrics(memory_logger):
    """Reproduce issue #1357: OpenRouter returns is_byok boolean in usage.

    Makes a real litellm.completion call via OpenRouter. The response includes
    `is_byok: true` in usage, which must be filtered out of metrics to avoid
    a 400 from the Braintrust API.
    """
    import os

    assert not memory_logger.pop()

    start = time.time()
    response = litellm.completion(
        model="openrouter/openai/gpt-4o-mini",
        messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
        max_tokens=10,
        api_key=os.environ.get("OPENROUTER_API_KEY", "fake-key"),
    )
    end = time.time()

    assert response
    assert response.choices[0].message.content

    spans = memory_logger.pop()
    assert len(spans) == 1
    metrics = spans[0]["metrics"]
    metadata = spans[0]["metadata"]
    assert "api_key" not in metadata
    assert metadata["provider"] == "openrouter"

    for key, value in metrics.items():
        assert not isinstance(value, bool)
    assert "is_byok" not in metrics


@pytest.mark.vcr
@sync_async
def test_litellm_rerank(memory_logger, is_async):
    assert not memory_logger.pop()

    start = time.time()
    response = _run(
        is_async,
        litellm.rerank,
        litellm.arerank,
        model=RERANK_MODEL,
        query=RERANK_QUERY,
        documents=RERANK_DOCUMENTS,
        top_n=2,
    )
    end = time.time()

    assert response
    assert response.results
    assert len(response.results) == 2
    # Paris should rank first for "capital of France".
    assert response.results[0]["index"] == 0

    spans = memory_logger.pop()
    assert len(spans) == 1
    span = spans[0]
    assert span["span_attributes"]["name"] == "Rerank"
    assert span["span_attributes"]["type"] == "llm"
    assert span["metadata"]["provider"] == "cohere"
    assert span["metadata"]["model"] == RERANK_MODEL
    assert span["metadata"]["top_n"] == 2
    assert span["metadata"]["document_count"] == 3
    assert span["input"] == {"query": RERANK_QUERY, "documents": RERANK_DOCUMENTS}

    assert isinstance(span["output"], list)
    assert len(span["output"]) == 2
    assert span["output"][0]["index"] == 0
    assert isinstance(span["output"][0]["relevance_score"], float)
    # Document payload must not leak into the span output.
    for entry in span["output"]:
        assert set(entry.keys()) == {"index", "relevance_score"}

    metrics = span["metrics"]
    assert metrics["start"] >= start
    assert metrics["end"] <= end
    assert metrics["end"] >= metrics["start"]
    assert metrics["duration"] >= 0
    # Cohere rerank bills in search_units.
    assert metrics.get("search_units") == 1


def test_litellm_parse_rerank_metrics_from_meta():
    """Unit-level sanity check for ``_parse_rerank_metrics``.

    LiteLLM rerank responses follow Cohere's shape: usage lives under
    ``meta.billed_units`` and ``meta.tokens``.  ``billed_units`` wins when
    both are present.
    """
    from braintrust.integrations.litellm.tracing import _parse_rerank_metrics

    response = {
        "meta": {
            "billed_units": {
                "input_tokens": 7,
                "output_tokens": 3,
                "search_units": 1,
            },
            "tokens": {
                "input_tokens": 999,
                "output_tokens": 999,
            },
        }
    }
    metrics = _parse_rerank_metrics(response)
    # billed_units overrides the matching fields in tokens.
    assert metrics["prompt_tokens"] == 7
    assert metrics["completion_tokens"] == 3
    assert metrics["search_units"] == 1
    # derived total when no total_tokens is reported.
    assert metrics["tokens"] == 10

    # meta.billed_units alone also yields a derived total.
    metrics2 = _parse_rerank_metrics({"meta": {"billed_units": {"search_units": 1}}})
    assert metrics2 == {"search_units": 1}


class TestAutoInstrumentLiteLLM:
    """Tests for auto_instrument() with LiteLLM."""

    def test_auto_instrument_litellm(self):
        """Test auto_instrument patches LiteLLM, creates spans, and uninstrument works."""
        verify_autoinstrument_script("test_auto_litellm.py")
