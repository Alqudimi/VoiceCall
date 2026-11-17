# syntax=docker/dockerfile:1
# =============================================================================
# Iraqi Voice Studio — rvc-platform runtime container (CPU wheels)
# GPU wheels are deliberately NOT baked in: `torch` ships as the CPU wheel so
# the image boots anywhere; GPU users add the CUDA variant per-machine with
#   `make env-gpu CUDA=cu128`
# (see the note in pyproject.toml [project.optional-dependencies]).
# Build:  docker build -t iraqi-voice-studio .
# Run:    docker run -p 8320:8320 -v rvc-data:/data iraqi-voice-studio
# =============================================================================
FROM python:3.15-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    RVC_PLATFORM_ROOT=/data

# ffmpeg: audio format conversions in the platform CLI/API
# git: needed by the model/assets tooling
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg git ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# uv — reproducible installs straight from uv.lock
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

WORKDIR /app

# ---- dependency layer (cached while uv.lock is untouched) ----
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-group dev --no-install-project

# ---- project layer ----
COPY src ./src
COPY configs ./configs
COPY README.md ./
RUN uv sync --frozen --no-group dev

# ---- runtime data volume: logs / runs / models / assets / data ----
VOLUME /data
EXPOSE 8320

# Port-level liveness (the HTTP root requires the auth token, so we probe TCP)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import socket; socket.create_connection(('127.0.0.1', 8320), timeout=5).close()" \
  || exit 1

ENTRYPOINT ["uv", "run", "--no-sync", "rvc-platform"]
CMD ["web", "--host", "0.0.0.0", "--port", "8320"]
