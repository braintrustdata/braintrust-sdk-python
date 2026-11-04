"""Effective privacy of the qualified LiveKit release, including prior setup."""

import logging
import os

from braintrust.span_customizer import SpanCustomizer


def opted_in(direction):
    return os.getenv(f"BRAINTRUST_CAPTURE_{direction}_AUDIO_ATTACHMENTS", "").strip().lower() in (
        "true",
        "1",
        "yes",
        "on",
    )


class Privacy:
    def __init__(self, session, record):
        from livekit.agents import get_job_context

        self.session = session
        self.job = get_job_context(required=False)
        self.restricted = isinstance(record, dict) and bool(record.get("redaction"))
        self._warned = False

    def allows_content(self):
        from livekit.agents.telemetry import traces, utils
        from livekit.agents.telemetry.pii import _PIIFilteringSpanProcessor

        if self.restricted:
            return False
        try:
            project = self.job is not None and (self.job._redaction_enabled or self.job.job.enable_redaction)
            restricted = project or bool(getattr(self.session, "_redaction_enabled", False))
            provider = traces.tracer._tracer_provider
            processor = getattr(provider, "_active_span_processor", None)
            processors = getattr(processor, "_span_processors", ())
            policies = [p._allow_pii for p in processors if isinstance(p, _PIIFilteringSpanProcessor)]
            # An installed provider policy is the effective explicit/env snapshot.
            # Without one, LiveKit's current environment/default governs.
            allows = all(policies) if policies else utils.allow_pii_from_env() is not False
            restricted = restricted or not allows
        except Exception:
            restricted = True
            if not self._warned:
                self._warned = True
                logging.getLogger("braintrust").warning("LiveKit privacy policy unavailable; withholding content")
        self.restricted = bool(restricted)
        return not self.restricted


class PrivacyCustomizer(SpanCustomizer):
    """Recheck queued content at the ordinary span customization boundary."""

    def __init__(self, policy):
        self.policy = policy

    def on_span_export(self, data):
        if self.policy.allows_content():
            return data
        for field in ("input", "output", "error"):
            data.pop(field, None)
        attributes = data.get("span_attributes", {})
        if attributes.get("type") == "tool":
            data["span_attributes"] = {**attributes, "name": "tool"}
        metadata = dict(data.get("metadata", {}))
        if "audio.segments" in metadata:
            metadata["audio.segments"] = {
                key: {
                    "state": "omitted",
                    "reason": "privacy_restricted",
                    **{field: segment[field] for field in ("start_offset_ms", "duration_ms") if field in segment},
                }
                for key, segment in metadata["audio.segments"].items()
            }
        for field in ("audio.participants", "audio.speakers", "audio.participant_ids", "audio.speaker_ids"):
            metadata.pop(field, None)
        if metadata:
            data["metadata"] = metadata
        return data
