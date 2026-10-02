#!/usr/bin/env bash
set -euo pipefail

export RAGIFY_GRPC_HOST="${RAGIFY_GRPC_HOST:-0.0.0.0}"
# Prefer an explicit RAGIFY_GRPC_PORT, then Cloud Run's injected $PORT
# (8080 there), then the conventional gRPC port for local container runs.
export RAGIFY_GRPC_PORT="${RAGIFY_GRPC_PORT:-${PORT:-50051}}"

# Embeddings run in-process via transformers; the model was baked in at build
# time and loads lazily on the first embedding call.
exec python -m src.grpc
