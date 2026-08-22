FROM python:3.11-slim

WORKDIR /app

COPY requirements-mlflow.txt requirements-worker.txt ./
# CPU-only torch: sentence-transformers depends on torch, and installing it
# unconstrained pulls the CUDA-enabled build (cuda-toolkit, nvidia-cublas,
# nvidia-cudnn, etc. -- several GB of GPU libraries this container never
# uses). Installing the CPU wheel first satisfies that dependency, so the
# next step reuses it instead of resolving a different (CUDA) build.
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir -r requirements-worker.txt

# Root-level judge-scoring modules, reused unmodified from the CLI tool.
COPY evaluator.py similarity.py prompts.py schemas.py dataset_eval.py ./
COPY src ./src

ENV PYTHONPATH=/app
ENV PYTHONUNBUFFERED=1

EXPOSE 9100

CMD ["python", "-m", "src.workers.run_worker"]
