"""Lossless audio sidecars for test cassettes; never used by SDK instrumentation."""

import base64
import copy
import shutil
import wave
from contextlib import contextmanager
from email.parser import BytesParser
from email.policy import default
from pathlib import Path
from tempfile import TemporaryDirectory

from vcr.persisters.filesystem import CassetteNotFoundError
from vcr.serialize import deserialize, serialize


def write_audio(path, data, *, pcm=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if pcm:
        # These OpenAI speech/realtime fixtures use mono 24 kHz PCM16.
        with wave.Wave_write(str(path)) as audio:
            audio.setparams((1, 2, 24000, 0, "NONE", "not compressed"))
            audio.writeframes(data)
    else:
        path.write_bytes(data)


def read_audio(directory, reference):
    path = directory / reference["audio_file"]
    with wave.open(str(path)) as audio:
        if reference.get("pcm"):
            assert (audio.getnchannels(), audio.getsampwidth(), audio.getframerate()) == (1, 2, 24000)
        expected_bytes = audio.getnframes() * audio.getnchannels() * audio.getsampwidth()
        data = audio.readframes(audio.getnframes())
        assert len(data) == expected_bytes, "Truncated WAV fixture"
    if not reference.get("pcm"):
        data = path.read_bytes()
    start = reference.get("offset_bytes", 0)
    length = reference.get("length_bytes", len(data))
    assert 0 <= start <= start + length <= len(data), "Audio reference exceeds fixture file"
    return data[start : start + length]


def publish_cassette(staged, target):
    """Replace a complete fixture set; restore old audio if manifest replacement fails.

    Re-recording is sequential. This is rollback for ordinary I/O failures,
    not a filesystem transaction resilient to process termination/power loss.
    """
    audio = target.with_suffix(".audio")
    replacement = staged.with_suffix(".audio")
    replacement.mkdir(exist_ok=True)
    backup = staged.parent / "previous.audio"
    had_audio = audio.exists()
    if had_audio:
        audio.replace(backup)
    installed = False
    try:
        replacement.replace(audio)
        installed = True
        staged.replace(target)
    except BaseException:
        if installed:
            shutil.rmtree(audio)
        if had_audio:
            backup.replace(audio)
        raise


@contextmanager
def staged_cassette(target):
    target.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".recording-", dir=target.parent) as directory:
        staged = Path(directory) / target.name
        yield staged
        publish_cassette(staged, target)


class AudioPersister:
    """Use ordinary VCR serialization after restoring exact HTTP body bytes."""

    @staticmethod
    def load_cassette(cassette_path, serializer):
        path = Path(cassette_path)
        if not path.is_file():
            raise CassetteNotFoundError()
        data = serializer.deserialize(path.read_text())
        for interaction in data["interactions"]:
            body = interaction["request"].get("body")
            if isinstance(body, dict) and "parts" in body:
                interaction["request"]["body"] = b"".join(
                    part.encode() if isinstance(part, str) else read_audio(path.parent, part) for part in body["parts"]
                )
            body = interaction["response"]["body"].get("string")
            if isinstance(body, dict) and "audio_file" in body:
                interaction["response"]["body"]["string"] = read_audio(path.parent, body)
        return deserialize(serializer.serialize(data), serializer)

    @staticmethod
    def save_cassette(cassette_path, cassette_dict, serializer):
        with staged_cassette(Path(cassette_path)) as staged:
            AudioPersister._save(staged, cassette_dict, serializer)

    @staticmethod
    def _save(path, cassette_dict, serializer):
        data = serializer.deserialize(serialize(copy.deepcopy(cassette_dict), serializer))
        directory = path.with_suffix(".audio")
        for index, interaction in enumerate(data["interactions"]):
            request = interaction["request"]
            headers = {key.lower(): value for key, value in request["headers"].items()}
            content_type = headers.get("content-type", [""])[0]
            if "multipart/" in content_type:
                body = request["body"]
                message = BytesParser(policy=default).parsebytes(
                    f"Content-Type: {content_type}\r\n\r\n".encode() + body
                )
                parts, remaining = [], body
                for number, part in enumerate(message.iter_parts()):
                    if not part.get_filename():
                        continue
                    payload = part.get_payload(decode=True)
                    # Preserve multipart headers, boundaries and text verbatim.
                    prefix, found, remaining = remaining.partition(payload)
                    assert found, "Multipart audio payload missing"
                    name = directory / f"request-{index}-{number}.wav"
                    write_audio(name, payload)
                    parts.extend([prefix.decode(), {"audio_file": f"{directory.name}/{name.name}"}])
                if parts:
                    request["body"] = {"parts": [*parts, remaining.decode()]}
            response = interaction["response"]
            headers = {key.lower(): value for key, value in response["headers"].items()}
            content_type = headers.get("content-type", [""])[0]
            if "audio/" in content_type or "application/octet-stream" in content_type:
                # The test requests PCM explicitly; do not reinterpret arbitrary codecs.
                import json

                assert json.loads(request["body"])["response_format"] == "pcm"
                name = directory / f"response-{index}.wav"
                write_audio(name, response["body"]["string"], pcm=True)
                response["body"]["string"] = {"audio_file": f"{directory.name}/{name.name}", "pcm": True}
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(serializer.serialize(data))


def save_websocket(path, endpoint, events):
    with staged_cassette(path) as staged:
        _save_websocket(staged, endpoint, events)


def _save_websocket(path, endpoint, events):
    """Combine chunks into playable files, retaining each chunk's exact range."""
    import json

    events = copy.deepcopy(events)
    directory = path.with_suffix(".audio")
    streams = {}
    for event in events:
        message = event["message"]
        kind = message.get("type")
        field = (
            "audio"
            if kind == "input_audio_buffer.append"
            else "delta"
            if kind == "response.output_audio.delta"
            else None
        )
        if field is None:
            continue
        key = "caller" if field == "audio" else message["response_id"]
        if key not in streams:
            name = "caller.wav" if key == "caller" else f"agent-{sum(k != 'caller' for k in streams):03d}.wav"
            streams[key] = (name, bytearray())
        name, pcm = streams[key]
        chunk = base64.b64decode(message[field], validate=True)
        message[field] = {
            "audio_file": f"{directory.name}/{name}",
            "pcm": True,
            "offset_bytes": len(pcm),
            "length_bytes": len(chunk),
        }
        pcm.extend(chunk)
    for name, pcm in streams.values():
        write_audio(directory / name, pcm, pcm=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"endpoint": endpoint, "events": events}, indent=2) + "\n")


def load_websocket(path):
    import json

    data = json.loads(path.read_text())
    for event in data["events"]:
        message = event["message"]
        for field in ("audio", "delta"):
            reference = message.get(field)
            if isinstance(reference, dict) and "audio_file" in reference:
                message[field] = base64.b64encode(read_audio(path.parent, reference)).decode()
    return data
