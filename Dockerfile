# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.11.30 AS uv


# Shared GPU dependencies
FROM nvidia/cuda:13.3.0-devel-ubuntu24.04 AS gpu-deps

COPY --from=uv /uv /bin/uv

ENV CUDA_HOME=/usr/local/cuda \
    UV_PYTHON_INSTALL_DIR=/opt/uv/python \
    PATH=/app/.venv/bin:${PATH}

RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        git \
        libgomp1 \
        libnuma1 \
        util-linux \
    && rm -rf /var/lib/apt/lists/*

RUN uv python install 3.11

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync \
        --python 3.11 \
        --locked \
        --no-dev \
        --link-mode copy

RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install \
        --python /app/.venv/bin/python \
        --reinstall \
        --no-deps \
        --link-mode copy \
        nvidia-cutlass-dsl-libs-cu13==4.5.2

COPY pyproject.toml uv.lock ./


# GPU server
FROM gpu-deps AS server

COPY sglang_recurrent sglang_recurrent

ENTRYPOINT ["uv", "run", "--locked", "--no-sync", "--no-dev", "-m", "sglang_recurrent.server"]
CMD ["--model", "models/Ouro-1.4B-Thinking"]


# GPU demo
FROM gpu-deps AS demo

COPY sglang_recurrent sglang_recurrent
COPY demo demo

ENTRYPOINT ["bash", "demo/run.sh"]


# CPU evaluation client
FROM ghcr.io/astral-sh/uv:0.11.30-python3.11-trixie-slim AS eval

WORKDIR /app

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync \
        --locked \
        --only-group eval \
        --link-mode copy

COPY pyproject.toml uv.lock ./
COPY evaluations evaluations

ENTRYPOINT ["uv", "run", "--locked", "--no-sync", "--only-group", "eval", "-m", "evaluations"]
