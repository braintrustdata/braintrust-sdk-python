"""Public audio recording configuration shared by voice integrations.

These options choose segmentation and encoding. Capture remains opt-in through
an integration's capture options or the dedicated USER/AGENT audio environment
variables. Importing configuration does not load PyAV or a voice framework.
"""

from ._audio.config import AudioEncoding as AudioEncoding
from ._audio.config import AudioRecordingOptions as AudioRecordingOptions


__all__ = ["AudioEncoding", "AudioRecordingOptions"]
