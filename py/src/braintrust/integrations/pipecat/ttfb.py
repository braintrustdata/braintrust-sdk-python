"""Associate reported TTFB only with an unambiguous live processor operation."""


class TTFBRouter:
    def __init__(self, fallback):
        self.fallback = fallback
        self.active = {}
        self.unmatched = {}

    def start(self, key, processor, log):
        # Repeated starts without a matching end cannot identify one operation.
        if key in self.active:
            processor = None
        self.active[key] = (processor, log, {})

    def end(self, key):
        self.active.pop(key, None)

    def clear(self):
        self.active.clear()

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
        key, log, state = self.owner(metric, source)
        name = getattr(metric, "processor", None)
        values = state.setdefault("values", [])
        if len(values) >= 32:
            state["omitted"] = state.get("omitted", 0) + 1
            log(metadata={"braintrust.ttfb.omitted": state["omitted"]})
        else:
            payload = (
                metric.model_dump(mode="json")
                if hasattr(metric, "model_dump")
                else {"processor": name, "value": metric.value}
            )
            values.append(payload)
            log(metadata={"pipecat.ttfb": list(values)})
        return key
