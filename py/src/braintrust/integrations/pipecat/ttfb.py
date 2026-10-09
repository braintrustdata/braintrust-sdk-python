"""Route native measurements by request identity or an unambiguous live operation."""

from collections import OrderedDict, deque


class TTFBRouter:
    def __init__(self, fallback):
        self.fallback = fallback
        self.active = {}
        self.unmatched = {}
        self.pending = deque()
        self.completed = OrderedDict()

    def start(self, key, processor, log):
        self.completed.pop(key, None)
        # Repeated starts without a matching end cannot identify one operation.
        if key in self.active:
            processor = None
        self.active[key] = (processor, log, {})
        pending, self.pending = self.pending, deque()
        for metric, source, request in pending:
            if request == key:
                self.capture_request(metric, source, request)
            else:
                self.pending.append((metric, source, request))

    def end(self, key):
        operation = self.active.pop(key, None)
        if operation and isinstance(key, tuple) and key[0] == "tts":
            self.completed[key] = operation
            if len(self.completed) > 64:
                self.completed.popitem(last=False)

    def clear(self):
        while self.pending:
            metric, _, _ = self.pending.popleft()
            self._capture_request(metric, (None, self.fallback, self.unmatched))
        self.active.clear()
        self.completed.clear()

    def capture_request(self, metric, source, key):
        """Use emission-time identity, including before start or after stop."""
        operation = self.active.get(key) or self.completed.get(key)
        if operation:
            processor, log, state = operation
            owner = (
                (key, log, state)
                if processor is source and metric.processor == source.name
                else (None, self.fallback, self.unmatched)
            )
            self._capture_request(metric, owner)
        else:
            if len(self.pending) >= 256:
                oldest, _, _ = self.pending.popleft()
                self._capture_request(oldest, (None, self.fallback, self.unmatched))
            self.pending.append((metric, source, key))

    def _capture_request(self, metric, owner):
        if type(metric).__name__ == "TTFBMetricsData":
            self._capture_ttfb(metric, owner)
        else:
            self._capture_measurement(metric, owner)

    def owner(self, metric, source, *, operation=None):
        name = getattr(metric, "processor", None)
        matches = [
            (key, log, state)
            for key, (processor, log, state) in self.active.items()
            if processor is source
            and name is not None
            and getattr(processor, "name", None) == name
            and (operation is None or key == operation)
        ]
        if len(matches) == 1:
            return matches[0]
        return None, self.fallback, self.unmatched

    def capture(self, metric, source):
        return self._capture_ttfb(metric, self.owner(metric, source))

    def _capture_ttfb(self, metric, owner):
        key, log, state = owner
        values = state.setdefault("values", [])
        if len(values) >= 32:
            state["omitted"] = state.get("omitted", 0) + 1
            log(metadata={"braintrust.ttfb.omitted": state["omitted"]})
        else:
            payload = (
                metric.model_dump(mode="json")
                if hasattr(metric, "model_dump")
                else {"processor": getattr(metric, "processor", None), "value": metric.value}
            )
            values.append(payload)
            log(metadata={"contrib.pipecat.ttfb": list(values)})
        return key

    def capture_measurement(self, metric, source):
        """Retain native measurements without a frame/routing envelope."""
        self._capture_measurement(metric, self.owner(metric, source))

    def _capture_measurement(self, metric, owner):
        _, log, state = owner
        values = state.setdefault("measurements", [])
        if len(values) >= 32:
            state["measurements_omitted"] = state.get("measurements_omitted", 0) + 1
            log(metadata={"braintrust.measurements.omitted": state["measurements_omitted"]})
            return
        values.append({"type": type(metric).__name__, **metric.model_dump(mode="json")})
        log(metadata={"contrib.pipecat.measurements": list(values)})
