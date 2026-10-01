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
