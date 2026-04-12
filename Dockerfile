FROM ghcr.io/meta-pytorch/openenv-base:latest

WORKDIR /app

RUN apt-get update && \
    apt-get install -y --no-install-recommends git && \
    rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir fastapi pydantic
RUN pip install --no-cache-dir "openenv-core[core] @ git+https://github.com/meta-pytorch/OpenEnv.git@v0.2.1"

# Bust cache on every push
ENV BUILD_VERSION=v6

# Copy all code AFTER cache bust
COPY . /app/enterprise_guardian

ENV PYTHONPATH=/app

CMD ["uvicorn", "enterprise_guardian.server.app:app", "--host", "0.0.0.0", "--port", "7860"]
