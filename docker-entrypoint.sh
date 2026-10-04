#!/usr/bin/env bash
set -euo pipefail

export RAGIFY_GRPC_HOST="${RAGIFY_GRPC_HOST:-0.0.0.0}"
# Cloud Run injects $PORT and probes exactly that port, so it has to win over a
# RAGIFY_GRPC_PORT carried over from local/compose runs. A stale 50051 pin here
# makes the server bind 50051 while Cloud Run probes 8080, which fails the
# startup probe with DEADLINE_EXCEEDED.
export RAGIFY_GRPC_PORT="${PORT:-${RAGIFY_GRPC_PORT:-50051}}"

echo "[ragify-entrypoint] gRPC server binding ${RAGIFY_GRPC_HOST}:${RAGIFY_GRPC_PORT}"

# Embeddings run in-process via transformers; the model was baked in at build
# time and loads lazily on the first embedding call.
exec python -m src.grpc
