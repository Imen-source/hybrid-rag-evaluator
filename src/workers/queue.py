"""Redis/RQ queue access shared by the API (enqueue) and the worker (consume)."""

from __future__ import annotations

import os

from redis import Redis
from rq import Queue

EVAL_QUEUE_NAME = "eval"


def get_redis_connection() -> Redis:
    # Read the env var at call time, not import time: tests point this at a
    # dynamically-provisioned testcontainers Redis after this module is
    # already imported.
    redis_url = os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    return Redis.from_url(redis_url)


def get_eval_queue() -> Queue:
    return Queue(EVAL_QUEUE_NAME, connection=get_redis_connection())
