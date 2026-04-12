FROM python:3.11-slim

WORKDIR /app
COPY . /app/enterprise_guardian
# Move pyproject.toml to root to allow pip install
RUN mv /app/enterprise_guardian/pyproject.toml /app/pyproject.toml

RUN pip install --no-cache-dir -e .[server]
RUN pip install --no-cache-dir uvicorn openenv-core[core] fastapi

ENV PYTHONPATH=/app

CMD ["uvicorn", "enterprise_guardian.server.app:app", "--host", "0.0.0.0", "--port", "8000"]
