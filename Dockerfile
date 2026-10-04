FROM python:3.12-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Runtime-only subset of requirements.txt (see the header of that file).
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

# torch comes from the CPU-only index. The default PyPI wheel is the CUDA build
# and drags in several GB of NVIDIA libraries this image never uses. Installed
# after the pinned list so torch's other dependencies are already satisfied and
# only the wheel itself has to come from the PyTorch index.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch

# tiktoken fetches its BPE files on first use. Warming the cache at build time
# keeps the container from needing network egress at runtime.
ENV TIKTOKEN_CACHE_DIR=/opt/tiktoken
RUN python -c "import tiktoken; tiktoken.get_encoding('cl100k_base')"

# Embeddings run in-process, so the model is baked in at build time and the
# container needs no model server and no network access at runtime.
ARG EMBED_MODEL_HF=BAAI/bge-small-en-v1.5
ENV EMBED_BACKEND=transformers
ENV EMBED_MODEL_HF=${EMBED_MODEL_HF}
RUN python -c "import os; from huggingface_hub import snapshot_download; snapshot_download(os.environ['EMBED_MODEL_HF'])"

COPY src ./src
COPY docker-entrypoint.sh /usr/local/bin/ragify-entrypoint
RUN chmod +x /usr/local/bin/ragify-entrypoint

ENV RAGIFY_GRPC_HOST=0.0.0.0
# Plain local runs default to 50051; Cloud Run ignores EXPOSE, injects $PORT
# (8080), and the entrypoint gives PORT precedence.
EXPOSE 50051 8080

CMD ["ragify-entrypoint"]
