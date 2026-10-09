"""Per-pipeline reversible method and callback installation."""

import inspect
import logging
from contextvars import ContextVar
from functools import wraps


class Hooks:
    def __init__(self):
        self.methods = []
        self.handlers = []
        self.invocations = {}

    def original(self, target, name):
        original = getattr(target, name)
        if inspect.isasyncgenfunction(original):
            return original
        state = self.invocations.setdefault((id(target), name), ContextVar(name, default=None))

        def invoke(*args, **kwargs):
            current = state.get()
            if current is not None:
                current["called"] = True
            try:
                result = original(*args, **kwargs)
            except Exception as error:
                if current is not None:
                    current["error"] = error
                raise
            if current is not None:
                current["result"] = result
            return result

        @wraps(original)
        async def invoke_async(*args, **kwargs):
            current = state.get()
            if current is not None:
                current["called"] = True
            try:
                result = await original(*args, **kwargs)
            except Exception as error:
                if current is not None:
                    current["error"] = error
                raise
            if current is not None:
                current["result"] = result
            return result

        return invoke_async if inspect.iscoroutinefunction(original) else invoke

    def guard(self, target, name, replacement):
        state = self.invocations.get((id(target), name))
        if state is None or inspect.isasyncgenfunction(replacement):
            return replacement
        original = getattr(target, name)
        warned = False

        def failure(current):
            nonlocal warned
            if "error" in current:
                raise current["error"]
            if not warned:
                logging.getLogger(__name__).warning("Pipecat observation failed in %s", name)
                warned = True

        @wraps(replacement)
        async def guarded_async(*args, **kwargs):
            current = {}
            token = state.set(current)
            try:
                return await replacement(*args, **kwargs)
            except Exception:
                failure(current)
                if current.get("called"):
                    return current.get("result")
                return await original(*args, **kwargs)
            finally:
                state.reset(token)

        @wraps(replacement)
        def guarded(*args, **kwargs):
            current = {}
            token = state.set(current)
            try:
                return replacement(*args, **kwargs)
            except Exception:
                failure(current)
                if current.get("called"):
                    return current.get("result")
                return original(*args, **kwargs)
            finally:
                state.reset(token)

        return guarded_async if inspect.iscoroutinefunction(replacement) else guarded

    def set(self, target, name, replacement):
        original = target.__dict__.get(name)
        present = name in target.__dict__
        replacement = self.guard(target, name, replacement)
        setattr(target, name, replacement)
        self.methods.append((target, name, replacement, original, present))

    def remove(self, target, name):
        """Restore a replaced object's method without retaining it until shutdown."""
        for index in range(len(self.methods) - 1, -1, -1):
            entry = self.methods[index]
            if entry[0] is target and entry[1] == name:
                _, _, replacement, original, present = self.methods.pop(index)
                if target.__dict__.get(name) is replacement:
                    if present:
                        setattr(target, name, original)
                    else:
                        delattr(target, name)
                self.invocations.pop((id(target), name), None)
                return

    def event(self, target, name, handler):
        target.add_event_handler(name, handler)
        self.handlers.append((target, name, handler))

    def close(self):
        for target, name, handler in reversed(self.handlers):
            target.remove_event_handler(name, handler)
        self.handlers.clear()
        for target, name, replacement, original, present in reversed(self.methods):
            if target.__dict__.get(name) is replacement:
                if present:
                    setattr(target, name, original)
                else:
                    delattr(target, name)
        self.methods.clear()
        self.invocations.clear()
