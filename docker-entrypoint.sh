#!/usr/bin/env bash
set -euo pipefail

export OLLAMA_HOST="${OLLAMA_HOST:-http://127.0.0.1:11434}"
export RAGIFY_GRPC_HOST="${RAGIFY_GRPC_HOST:-0.0.0.0}"
# Cloud Run injects $PORT; keep 8080 for local container runs.
export RAGIFY_GRPC_PORT="${PORT:-8080}"

# Start Ollama in the background and wait until it accepts requests. The
# baked-in model is loaded lazily on the first embedding call.
ollama serve &
OLLAMA_PID=$!
cleanup() {
  kill "$OLLAMA_PID" 2>/dev/null || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  if ollama list >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

exec python -m src.grpc
