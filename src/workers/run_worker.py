"""Eval worker process entrypoint.

Starts the Prometheus metrics HTTP server once (see metrics.py for why this
requires a non-forking worker), then runs an RQ SimpleWorker against the
eval queue -- jobs execute in this same process rather than in a forked
child, so metrics recorded in tasks.py are visible on this process's
/metrics endpoint.
"""

from __future__ import annotations

import os

from rq.worker import SimpleWorker

from src.workers.metrics import start_metrics_server
from src.workers.queue import EVAL_QUEUE_NAME, get_eval_queue


def main() -> None:
    port = int(os.environ.get("WORKER_METRICS_PORT", "9100"))
    start_metrics_server(port)

    queue = get_eval_queue()
    worker = SimpleWorker([queue], connection=queue.connection)
    print(f"Eval worker started. Metrics on :{port}/metrics. Listening on queue '{EVAL_QUEUE_NAME}'.")
    worker.work()


if __name__ == "__main__":
    main()
