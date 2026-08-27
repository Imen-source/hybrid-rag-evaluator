FROM python:3.11-slim

WORKDIR /app

COPY requirements-api.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

# dataset_eval.py (+ its own dependency, root schemas.py) is needed by
# src/workers/regression.py, imported from the /eval/runs/{id}/compare
# route. Pure pydantic, no heavy judge deps (evaluator.py/similarity.py) --
# those stay worker-only, unlike in infra/worker.Dockerfile's equivalent COPY.
COPY schemas.py dataset_eval.py ./
COPY src ./src
COPY alembic ./alembic
COPY alembic.ini ./

EXPOSE 8000

CMD ["sh", "-c", "alembic upgrade head && uvicorn src.api.main:app --host 0.0.0.0 --port 8000"]
