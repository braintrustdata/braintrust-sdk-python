"""Exercise persistence with actual recorded traffic, without recontacting providers."""

import copy
from pathlib import Path

import pytest
from vcr.serializers import yamlserializer

from ._test_audio_cassettes import AudioPersister, load_websocket, read_audio, save_websocket


def test_audio_sidecars_round_trip_real_cassettes(tmp_path, vcr_cassette_dir):
    source = Path(vcr_cassette_dir)
    http_name = "test_cascade_voice_conversation.yaml"
    requests, responses = AudioPersister.load_cassette(source / http_name, yamlserializer)
    original = copy.deepcopy(responses)
    AudioPersister.save_cassette(tmp_path / http_name, {"requests": requests, "responses": responses}, yamlserializer)
    loaded_requests, loaded_responses = AudioPersister.load_cassette(tmp_path / http_name, yamlserializer)
    assert [r._to_dict() for r in loaded_requests] == [r._to_dict() for r in requests]
    assert loaded_responses == original == responses
    assert "!!binary" not in (tmp_path / http_name).read_text()

    socket_name = "test_realtime_voice_conversation.json"
    recorded = load_websocket(source / socket_name)
    original = copy.deepcopy(recorded)
    save_websocket(tmp_path / socket_name, recorded["endpoint"], recorded["events"])
    assert load_websocket(tmp_path / socket_name) == original == recorded
    sidecars = list(tmp_path.glob("*.audio/*.wav"))
    assert sidecars
    assert all(path.read_bytes().startswith(b"RIFF") for path in sidecars)
    for pattern, pcm in (("response-*.wav", True), ("request-*.wav", False)):
        audio = next(tmp_path.glob(f"test_cascade_voice_conversation.audio/{pattern}"))
        audio.write_bytes(audio.read_bytes()[:-4800])
        with pytest.raises(AssertionError, match="Truncated WAV"):
            read_audio(audio.parent, {"audio_file": audio.name, "pcm": pcm})
    # Missing/truncated files must fail replay, never silently substitute audio.
    caller = tmp_path / "test_realtime_voice_conversation.audio/caller.wav"
    caller.unlink()
    with pytest.raises(FileNotFoundError):
        load_websocket(tmp_path / socket_name)


@pytest.mark.parametrize("kind", ["http", "websocket"])
@pytest.mark.parametrize("failure_at", ["audio", "manifest"])
def test_failed_save_preserves_previous_fixture(tmp_path, vcr_cassette_dir, monkeypatch, kind, failure_at):
    from . import _test_audio_cassettes as storage

    source = Path(vcr_cassette_dir)
    if kind == "http":
        path = tmp_path / "test_cascade_voice_conversation.yaml"
        requests, responses = AudioPersister.load_cassette(source / path.name, yamlserializer)

        def save():
            AudioPersister.save_cassette(path, {"requests": requests, "responses": responses}, yamlserializer)
    else:
        path = tmp_path / "test_realtime_voice_conversation.json"
        data = load_websocket(source / path.name)

        def save():
            save_websocket(path, data["endpoint"], data["events"])

    save()
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    original_write = storage.write_audio

    def failed_write(destination, data, **kwargs):
        original_write(destination, data, **kwargs)
        destination.write_bytes(b"partial write")
        raise OSError("disk write failed")

    original_replace = Path.replace

    def failed_replace(source, target):
        if Path(target) == path:
            raise OSError("disk write failed")
        return original_replace(source, target)

    with monkeypatch.context() as patch:
        if failure_at == "audio":
            patch.setattr(storage, "write_audio", failed_write)
        else:
            patch.setattr(Path, "replace", failed_replace)
        with pytest.raises(OSError, match="disk write failed"):
            save()
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


@pytest.mark.parametrize("validated", [False, True])
def test_http_rerecord_promotes_only_validated_conversation(tmp_path, vcr_cassette_dir, validated):
    from types import SimpleNamespace

    import vcr

    from .test_voice_pipeline import vcr_cassette_name

    name = "test_cascade_voice_conversation"
    target = tmp_path / f"{name}.yaml"
    requests, responses = AudioPersister.load_cassette(Path(vcr_cassette_dir) / target.name, yamlserializer)
    AudioPersister.save_cassette(target, {"requests": requests, "responses": responses}, yamlserializer)
    target.write_text("# previous recording\n" + target.read_text())
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    recorder = vcr.VCR(record_mode="all")
    recorder.register_persister(AudioPersister)
    request = SimpleNamespace(node=SimpleNamespace(originalname=name))
    fixture = vcr_cassette_name.__wrapped__(request, str(tmp_path), recorder)
    with recorder.use_cassette(next(fixture)) as cassette:
        assert not cassette.requests, "re-recording must not append old interactions"
        for req, response in zip(requests, responses):
            cassette.append(req, response)
        request.node.voice_recording_validated = validated
    with pytest.raises(StopIteration):
        next(fixture)
    after = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    if validated:
        assert not target.read_text().startswith("# previous recording")
        loaded_requests, loaded_responses = AudioPersister.load_cassette(target, yamlserializer)
        assert len(loaded_requests) == len(requests)
        assert loaded_responses == responses
    else:
        assert after == before
