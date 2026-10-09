"""Internal audio API shared by integrations; codecs remain optional and lazy."""

from .alignment import InputRanges
from .attachments import UploadFailed, prepare_recording, upload_recording
from .budget import source_budget
from .export import AlignmentPublisher, segment_descriptor
from .jobs import RecordingJobs
from .options import RecordingOptions
from .recording import encode_audio
from .segments import SegmentedRecording
from .timeline import ClipTimeline, ms_to_samples, pcm_bytes_to_ms, samples_to_ms
from .worker import RecordingBusy, encode_in_worker


__all__ = [
    "AlignmentPublisher",
    "ClipTimeline",
    "InputRanges",
    "RecordingBusy",
    "RecordingJobs",
    "RecordingOptions",
    "SegmentedRecording",
    "UploadFailed",
    "encode_audio",
    "encode_in_worker",
    "ms_to_samples",
    "pcm_bytes_to_ms",
    "prepare_recording",
    "samples_to_ms",
    "segment_descriptor",
    "source_budget",
    "upload_recording",
]
