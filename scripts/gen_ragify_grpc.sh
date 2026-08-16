#!/usr/bin/env bash
# Regenerates gRPC stubs from src/ragify/grpc/ragify.proto for both the
# ragify-rag server and the backend's ragify_client library.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PROTO_DIR="$ROOT/src/grpc"
CLIENT_DIR="$ROOT/../api-python/src/ragify_client/protos"

mkdir -p "$CLIENT_DIR"

python -m grpc_tools.protoc -I "$PROTO_DIR" \
    --python_out="$PROTO_DIR" \
    --grpc_python_out="$PROTO_DIR" \
    "$PROTO_DIR/ragify.proto"

python -m grpc_tools.protoc -I "$PROTO_DIR" \
    --python_out="$CLIENT_DIR" \
    --grpc_python_out="$CLIENT_DIR" \
    "$PROTO_DIR/ragify.proto"

# Both copies live inside packages, so the pb2 import must be package-relative.
sed -i 's/^import ragify_pb2 as ragify__pb2$/from . import ragify_pb2 as ragify__pb2/' \
    "$PROTO_DIR/ragify_pb2_grpc.py" \
    "$CLIENT_DIR/ragify_pb2_grpc.py"

echo "Regenerated gRPC stubs in:"
echo "  $PROTO_DIR"
echo "  $CLIENT_DIR"
