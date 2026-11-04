"""One session-output observation point, with staged/accepted/played ownership."""

import math
from collections import deque
from dataclasses import dataclass, field
from fractions import Fraction

from livekit.agents.voice.io import AudioOutput, AudioOutputCapabilities

from .compat import detach_output


NS = 1_000_000_000


@dataclass
class Frame:
    offset: Fraction
    duration: Fraction
    rate: int
    channels: int
    data: bytes | None
    accepted: bool = False


@dataclass
class Run:
    parent: object
    queued_ns: int
    frames: list[Frame] = field(default_factory=list)
    duration: Fraction = Fraction(0)
    committed: Fraction = Fraction(0)
    progresses: deque = field(default_factory=deque)
    started: int | None = None
    played_end: int | None = None
    paused: bool = False
    finished: object = None
    capturing: int = 0
    speech: object = None
    selections: dict = field(default_factory=dict)
    finalized: bool = False
    sample_clock: int = 1


class PlaybackTap(AudioOutput):
    def __init__(self, output, owner):
        super().__init__(
            label="Braintrust recording",
            capabilities=AudioOutputCapabilities(pause=True),
            next_in_chain=output,
            sample_rate=output.sample_rate,
        )
        self.owner = owner
        self.original_output = output
        self.runs: deque[Run] = deque()
        self.current: Run | None = None
        self.on("playback_started", self._started)
        self.on("playback_progressed", self._progressed)
        self.on("playback_finished", self._finished)

    def _event_run(self):
        return next((run for run in self.runs if run.finished is None), None)

    async def capture_frame(self, frame):
        if self.current is None:
            self.current = Run(self.owner.agent_parent(), self.owner.audio.elapsed_ns())
            from .tracing import _SYNTHESIS

            self.current.synthesis = _SYNTHESIS.get()
            self.runs.append(self.current)
        run = self.current
        run.sample_clock = math.lcm(run.sample_clock, frame.sample_rate)
        duration = Fraction(frame.samples_per_channel, frame.sample_rate)
        staged = Frame(
            run.duration,
            duration,
            frame.sample_rate,
            frame.num_channels,
            bytes(frame.data) if self.owner.capture_output() else None,
        )
        run.duration += duration
        run.frames.append(staged)
        run.capturing += 1
        try:
            # Copy before awaiting the sink. It may reuse the native frame.
            await super().capture_frame(frame)
            await self.next_in_chain.capture_frame(frame)
            staged.accepted = True
        except BaseException:
            staged.data = None
            raise
        finally:
            run.capturing -= 1
            self._commit(run)
            self._finalize(run)
            self.owner.advance()

    def flush(self):
        self.current = None
        super().flush()
        self.next_in_chain.flush()

    def clear_buffer(self):
        self.current = None
        self.next_in_chain.clear_buffer()
        super().clear_buffer()

    def pause(self):
        if (run := self._event_run()) is not None:
            run.paused = True
        return super().pause()

    def _started(self, event):
        if (run := self._event_run()) is not None and run.started is None:
            run.started = self.owner.audio.from_wall(event.created_at)

    def _progressed(self, event):
        if (run := self._event_run()) is not None:
            offset = Fraction(round(event.offset * run.sample_clock), run.sample_clock)
            end = Fraction(round((event.offset + event.duration) * run.sample_clock), run.sample_clock)
            if offset >= 0 and end > offset:
                run.progresses.append((offset, end, self.owner.audio.from_wall(event.started_at)))
                self._commit(run)
                self.owner.advance()

    def _finished(self, event):
        if (run := self._event_run()) is None:
            return
        run.finished = event
        position = Fraction(round(max(0, event.playback_position) * run.sample_clock), run.sample_clock)
        if position > run.committed and run.started is not None and not run.paused:
            # Endpoint fallback is justified only for established contiguous
            # playback. Explicit progress always wins for pause/resume.
            run.progresses.append((run.committed, position, run.started + round(run.committed * NS)))
        self._commit(run)
        self._finalize(run)
        self.owner.advance()

    def _commit(self, run):
        while run.progresses:
            begin, end, wall = run.progresses[0]
            begin = max(begin, run.committed)
            if begin >= end:
                run.progresses.popleft()
                continue
            progressed = False
            for frame in run.frames:
                frame_end = frame.offset + frame.duration
                if frame_end <= begin or frame.offset >= end:
                    continue
                if not frame.accepted:
                    return  # Progress can race the awaiting capture call.
                first, last = max(begin, frame.offset), min(end, frame_end)
                # Keep both boundaries on the frame's sample clock.
                left = round((first - frame.offset) * frame.rate)
                right = round((last - frame.offset) * frame.rate)
                if right <= left:
                    continue
                start_ns = wall + (frame.offset + Fraction(left, frame.rate) - run.progresses[0][0]) * NS
                duration_ns = Fraction((right - left) * NS, frame.rate)
                if frame.data is not None and self.owner.capture_output():
                    stride = frame.channels * 2
                    self.owner.audio.record(
                        "agent-output",
                        frame.data[left * stride : right * stride],
                        sample_rate_hz=frame.rate,
                        channels=frame.channels,
                        offset_ns=start_ns,
                    )
                self.owner.played(run, start_ns, start_ns + duration_ns)
                run.played_end = round(start_ns + duration_ns)
                run.committed = last
                begin = last
                progressed = True
            if begin < end:
                # A future capture may fill the rest of this progress event.
                if not progressed:
                    return
                return
            run.progresses.popleft()
            # Release committed PCM once callbacks no longer need it.
            run.frames = [frame for frame in run.frames if frame.offset + frame.duration > run.committed]

    def _finalize(self, run):
        if run.finished is None or run.capturing or run.finalized:
            return
        run.finalized = True
        self._unavailable(run)
        self.owner.end_playback(run)
        run.frames.clear()
        run.progresses.clear()
        if run in self.runs:
            self.runs.remove(run)
        if self.current is run:
            self.current = None

    def _unavailable(self, run):
        if "AGENT" not in self.owner.directions:
            return
        for begin, end, wall in run.progresses:
            first = max(begin, run.committed)
            if first < end:
                self.owner.audio.unavailable(
                    wall + (first - begin) * NS, wall + (end - begin) * NS, "playback_capture_unavailable"
                )
        position = Fraction(round(getattr(run.finished, "playback_position", 0) * run.sample_clock), run.sample_clock)
        if not run.progresses and position > run.committed and (run.started is None or run.paused):
            self.owner.audio.unavailable(
                run.played_end or run.queued_ns, self.owner.audio.elapsed_ns(), "playback_timing_unavailable"
            )

    def watermark(self):
        holds = [run.played_end if run.played_end is not None else run.queued_ns for run in self.runs]
        return min(holds) if holds else self.owner.audio.elapsed_ns()

    def discard_private(self):
        for run in self.runs:
            for frame in run.frames:
                frame.data = None

    def close(self):
        for run in tuple(self.runs):
            # Teardown already settled native playback. No fabricated endpoint.
            self._unavailable(run)
            if run.finished is None and run.committed < run.duration and "AGENT" in self.owner.directions:
                self.owner.audio.unavailable(
                    run.played_end or run.queued_ns, self.owner.audio.elapsed_ns(), "playback_unsettled"
                )
            self.owner.end_playback(run)
            run.frames.clear()
        self.runs.clear()
        self.current = None
        detach_output(self)
        if self.next_in_chain is not self.original_output:
            # AudioOutput adds a native leaf proxy in its constructor. Detach
            # the proxy we created as well as our own observation listeners.
            detach_output(self.next_in_chain)
        self.off("playback_started", self._started)
        self.off("playback_progressed", self._progressed)
        self.off("playback_finished", self._finished)
        self.owner = None
        self.original_output = None
        self._next_in_chain = None
