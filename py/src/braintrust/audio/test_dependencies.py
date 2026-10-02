"""The shared capture path must work without optional codecs or frameworks."""

import subprocess
import sys


def test_capture_and_disabled_encode_without_optional_packages():
    code = """
import importlib.abc
import sys
import braintrust

class BlockOptional(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"pipecat", "numpy", "soundfile"}:
            raise AssertionError("Unexpected optional import: " + fullname)

sys.meta_path.insert(0, BlockOptional())
from braintrust.audio.recording import CallRecording
from braintrust.audio.alignment import InputRanges
from braintrust.audio import worker, RecordingOptions
from braintrust.audio.segments import SegmentedRecording
import asyncio
asyncio.run(SegmentedRecording(enabled=False, options=RecordingOptions()).finish())
assert worker._executor is None
recording = CallRecording()
interval = recording.capture(0, b"\\0\\0" * 480, 24000, 1, observed_ns=0)
ranges = InputRanges()
ranges.append(960, interval)
assert ranges.drain() == [[0, 480]]
recording.clear()
assert recording.encode() is None
assert CallRecording(enabled=False).encode() is None
assert worker._executor is None
"""
    subprocess.run([sys.executable, "-c", code], check=True, capture_output=True, text=True)
