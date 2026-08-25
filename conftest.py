"""Shared pytest fixtures: one Postgres + one Redis container per test
session, not per test file.

This matters beyond convenience: src/api/db.py builds its SQLAlchemy engine
once, at first import, from whatever DATABASE_URL is set at that moment. If
two test files each started their own Postgres container and set
DATABASE_URL independently, only the first-imported one would "win" for the
rest of the process -- the second file's tests would silently run against
the first file's database (and fail to find tables from migrations that
were only applied to the second container). Centralizing container startup
here, before any src.api.* module is imported, avoids that entirely.
"""

from __future__ import annotations

import os

import pytest

# Testcontainers' Ryuk cleanup sidecar fails to resolve its own exposed port
# under Docker Desktop on Windows (a known testcontainers-python/Windows
# interaction, not specific to this project), which aborts container startup
# before any test runs. Disabling it lets test containers start; the
# tradeoff is that if a test run is killed instead of exiting normally, the
# container will not be auto-removed and must be cleaned up manually
# (e.g. `docker ps` + `docker rm -f`).
os.environ.setdefault("TESTCONTAINERS_RYUK_DISABLED", "true")

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


@pytest.fixture(scope="session")
def postgres_db_url():
    from alembic.config import Config
    from testcontainers.postgres import PostgresContainer

    from alembic import command

    with PostgresContainer("postgres:16-alpine") as pg:
        db_url = pg.get_connection_url()
        os.environ["DATABASE_URL"] = db_url

        alembic_cfg = Config(os.path.join(REPO_ROOT, "alembic.ini"))
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)
        command.upgrade(alembic_cfg, "head")

        yield db_url


@pytest.fixture(scope="session")
def redis_db_url():
    from testcontainers.community.redis import RedisContainer

    with RedisContainer("redis:7-alpine") as redis_c:
        host = redis_c.get_container_host_ip()
        port = redis_c.get_exposed_port(6379)
        redis_url = f"redis://{host}:{port}/0"
        os.environ["REDIS_URL"] = redis_url
        yield redis_url


@pytest.fixture(scope="session")
def api_app(postgres_db_url):
    # Import only after DATABASE_URL is set (see postgres_db_url above), so
    # src/api/db.py builds its engine against the ephemeral test database.
    from src.api.main import app

    return app
