import pytest

from .hooks import Hooks


@pytest.mark.asyncio
@pytest.mark.parametrize("when", ["before", "after", "native"])
async def test_observation_failure_never_repeats_or_masks_native_call(when):
    class Target:
        calls = 0

        async def run(self, value):
            self.calls += 1
            if when == "native":
                raise ValueError("native error")
            return value * 2

    target, hooks = Target(), Hooks()
    original = hooks.original(target, "run")

    async def observed(value):
        if when == "before":
            raise RuntimeError("observer error")
        result = await original(value)
        raise RuntimeError("observer error after result")
        return result

    hooks.set(target, "run", observed)
    if when == "native":
        with pytest.raises(ValueError, match="native error"):
            await target.run(3)
    else:
        assert await target.run(3) == 6
    assert target.calls == 1
    hooks.close()
    assert "run" not in target.__dict__


@pytest.mark.asyncio
async def test_tts_request_tags_are_task_local_and_hooks_restore_on_cancellation():
    import asyncio
    from types import SimpleNamespace

    from pipecat.frames.frames import MetricsFrame  # pylint: disable=import-error
    from pipecat.metrics.metrics import TTSUsageMetricsData  # pylint: disable=import-error

    from .tts_metrics import TTSRequests

    both = asyncio.Event()
    entered = []
    tagged = []

    class Service:
        async def _push_tts_frames(self, frame):
            entered.append(frame.context_id)
            if len(entered) == 2:
                both.set()
            await both.wait()
            metric = MetricsFrame(data=[TTSUsageMetricsData(processor="tts", value=1)])
            await self.push_frame(metric)
            assert requests.take(metric, self) == ("tts", frame.context_id)
            assert requests.take(metric, self) is None  # Consumed once, never exported.
            detached = MetricsFrame(data=[])
            await asyncio.create_task(self.push_frame(detached))
            assert requests.take(detached, self) is None
            tagged.append(frame.context_id)
            if frame.context_id == "cancelled":
                raise asyncio.CancelledError()

        async def push_frame(self, frame):
            await asyncio.sleep(0)

    service, hooks = Service(), Hooks()
    requests = TTSRequests(hooks, [service])
    try:
        outcomes = await asyncio.gather(
            service._push_tts_frames(SimpleNamespace(context_id="first")),
            service._push_tts_frames(SimpleNamespace(context_id="cancelled")),
            return_exceptions=True,
        )
        assert sorted(tagged) == ["cancelled", "first"]
        assert outcomes[0] is None and isinstance(outcomes[1], asyncio.CancelledError)
        unassociated = MetricsFrame(data=[])
        await service.push_frame(unassociated)
        assert requests.take(unassociated, service) is None
    finally:
        hooks.close()
    assert "push_frame" not in service.__dict__ and "_push_tts_frames" not in service.__dict__
