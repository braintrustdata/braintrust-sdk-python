"""Native turn analyzer measurements, without inferred EOU timing."""

TURN_METRIC_TYPES = {"TurnMetricsData", "SmartTurnMetricsData"}
MAX_TURN_METRICS = 32


def log_turn_metric(span, state, metric):
    """Keep bounded predictions on their owner, preserving native units and identity."""
    predictions = state.setdefault("turn_metrics", [])
    if len(predictions) >= MAX_TURN_METRICS:
        state["turn_metrics_omitted"] = state.get("turn_metrics_omitted", 0) + 1
        span.log(metadata={"braintrust.turn_metrics.omitted": state["turn_metrics_omitted"]})
        return
    predictions.append(
        {
            "type": type(metric).__name__,
            **{
                name: getattr(metric, name)
                for name in ("processor", "is_complete", "probability", "e2e_processing_time_ms")
            },
        }
    )
    span.log(metadata={"pipecat.turn_metrics": list(predictions)})
