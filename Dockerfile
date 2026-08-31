FROM python:3.12-slim

WORKDIR /app

# Ollama is bundled so embeddings run in-process (localhost) with no extra
# service or VM. The model is pulled at build time and baked into the image.
ARG OLLAMA_VERSION=0.33.2
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl zstd \
    && rm -rf /var/lib/apt/lists/* \
    && curl -fsSL -o /tmp/ollama-install.sh https://ollama.com/install.sh \
    && OLLAMA_VERSION="${OLLAMA_VERSION}" sh /tmp/ollama-install.sh \
    && rm /tmp/ollama-install.sh

ARG EMBED_MODEL=qllama/bge-small-en-v1.5:latest
RUN set -eux; \
    ollama serve >/tmp/ollama-build.log 2>&1 & \
    OLLAMA_PID=$!; \
    for i in $(seq 1 30); do \
      if ollama list >/dev/null 2>&1; then break; fi; \
      sleep 2; \
    done; \
    ollama pull "$EMBED_MODEL"; \
    kill $OLLAMA_PID

# Trimmed lockfile: excludes torch/transformers/CUDA (reranker only, which is
# not part of the served graph; its imports are lazy).
COPY requirements-prod.txt ./
RUN pip install --no-cache-dir -r requirements-prod.txt

COPY src ./src
COPY docker-entrypoint.sh /usr/local/bin/ragify-entrypoint
RUN chmod +x /usr/local/bin/ragify-entrypoint

ENV EMBED_MODEL=qllama/bge-small-en-v1.5:latest
ENV RAGIFY_GRPC_HOST=0.0.0.0
EXPOSE 8080

CMD ["ragify-entrypoint"]
