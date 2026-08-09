# The index is built during the image build rather than committed, so it stays a
# derived artifact of corpus/ and cannot drift from it. Ingestion needs an
# OpenAI key to embed, which Render exposes to builds as a secret file.
#
# If the key is not available at build time the build still succeeds with an
# empty index, and the entrypoint ingests on first start instead. That fallback
# is safe only because ingestion is idempotent: when the image already carries a
# complete index, the startup pass compares content hashes, finds nothing to do
# and exits in about a second.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /srv

# Dependencies are installed before the source is copied so that editing code
# does not invalidate the layer that takes the longest to build.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Render mounts secret files at /etc/secrets. The key is read from there or from
# a build argument, and never written into a layer.
RUN --mount=type=secret,id=openai_api_key,required=false \
    set -eu; \
    if [ -f /run/secrets/openai_api_key ]; then \
        OPENAI_API_KEY="$(cat /run/secrets/openai_api_key)"; \
    elif [ -f /etc/secrets/OPENAI_API_KEY ]; then \
        OPENAI_API_KEY="$(cat /etc/secrets/OPENAI_API_KEY)"; \
    fi; \
    if [ -n "${OPENAI_API_KEY:-}" ]; then \
        export OPENAI_API_KEY; \
        python -m ingest.pipeline; \
    else \
        echo "No OpenAI key at build time; the index will be built on first start."; \
    fi

EXPOSE 8000
CMD ["./docker-entrypoint.sh"]
