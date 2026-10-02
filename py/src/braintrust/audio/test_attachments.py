"""Encoded audio stays budgeted for as long as the exporter retains it."""

import gc

import pytest

from .attachments import RecordingAttachment, encoded_budget, prepare_recording
from .worker import RecordingBusy


def test_encoded_budget_follows_attachment_lifetime(monkeypatch):
    previous = encoded_budget.used
    monkeypatch.setattr(encoded_budget, "maximum", previous + 4)
    attachment = RecordingAttachment(data=b"1234", filename="clip.wav", content_type="audio/wav")
    with pytest.raises(RecordingBusy):
        RecordingAttachment(data=b"x", filename="clip.wav", content_type="audio/wav")
    assert encoded_budget.used == previous + 4
    del attachment
    # Completed uploads must not wait for cyclic GC to return capacity.
    assert encoded_budget.used == previous
    gc.collect()


def test_prepare_recording_preserves_disabled_result():
    assert prepare_recording("call", lambda: None) is None
