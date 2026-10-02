"""Send nested traces and ordered score updates through the ingestion transport.

From py/: mise exec -- python ../examples/log_ingestion/demo.py
Add --overflow to exercise a signed overflow upload on a supporting server.
"""

import argparse
import json
import uuid

import braintrust
from braintrust.logger import _internal_get_global_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="sdk-839-ingestion-demo")
    parser.add_argument("--overflow", action="store_true")
    parser.add_argument(
        "--replay", action="store_true", help="Replay the same final feedback records to check server deduplication"
    )
    args = parser.parse_args()
    logger = braintrust.init_logger(project=args.project)
    run_id = str(uuid.uuid4())
    state = _internal_get_global_state()
    writer = state.global_bg_logger()
    writer.sync_flush = True
    if args.overflow:
        writer._max_request_size_override = 1024
        writer._max_request_size_result = None
    with logger.start_span(
        name="policy-aware-ingestion-demo", input={"run_id": run_id}, metadata={"issue": 839}
    ) as root:
        with root.start_span(name="child", input="hello") as child:
            child.log(output="world", scores={"quality": 1})
        root.log(
            output={"status": "delivered", "payload": "ü" * (2000 if args.overflow else 1)}, scores={"quality": 0}
        )
    braintrust.flush()
    # Two later updates must reach the same row in order, including after relogin.
    logger.log_feedback(id=root.id, scores={"quality": 0.5})
    braintrust.flush()
    braintrust.login(force_login=True)
    logger.log_feedback(id=root.id, scores={"quality": 1}, comment="Verified ordered score update after relogin")
    replay_records = writer.queue.drain_all(reserve=True) if args.replay else []
    if replay_records:
        writer.queue.restore(replay_records)
    braintrust.flush()
    if replay_records:
        for record in replay_records:
            writer.queue.put(record)
        braintrust.flush()
    print(
        json.dumps(
            {
                "run_id": run_id,
                "project_id": logger.id,
                "row_id": root.id,
                "trace_id": root.root_span_id,
                "link": root.link(),
                "pending_rows": writer.pending_count,
                "overflow_uploads": writer._overflow_upload_count,
                "replayed_feedback": bool(replay_records),
                "overflow_supported": (writer._max_request_size_result or {}).get("can_use_overflow"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
