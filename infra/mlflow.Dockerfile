FROM python:3.11-slim

WORKDIR /mlflow

COPY requirements-mlflow.txt ./
RUN pip install --no-cache-dir -r requirements-mlflow.txt

EXPOSE 5000

CMD ["mlflow", "server", \
     "--host", "0.0.0.0", \
     "--port", "5000", \
     "--backend-store-uri", "sqlite:////mlflow/data/mlflow.db", \
     "--default-artifact-root", "/mlflow/data/artifacts"]
