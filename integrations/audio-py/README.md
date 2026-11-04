# Braintrust audio

Install `braintrust-audio` alongside a core SDK containing the audio recording
contract to encode enabled session recordings as Opus in Ogg by default. The
package uses PyAV's bundled FFmpeg codecs. Each segment is an independent file
with exact playable duration.

Installing this extension does not enable recording. Capture requires the
framework adapter's audio opt-ins and privacy permission. Without the extension,
default recording warns once and uses raw PCM. Explicit Opus configuration
requires this extension. Codec and container are selected separately with
`AudioEncoding(codec="opus", container="ogg")`. PCM/WAV require no native codecs.

Audio uploads use `Attachment.upload()` directly. Span customizers and masking
apply to the trace records published afterward; removing an attachment reference
does not prevent or undo its upload.

Before publishing this distribution, set its minimum core dependency to the
release containing the new audio contract. The publish workflow defaults to a
dry run; configure the PyPI trusted publisher before the first release.
