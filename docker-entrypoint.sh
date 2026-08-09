#!/bin/sh
# Ensure an index exists, then serve.
#
# The ingestion pass is unconditional because it is idempotent and cheap when
# there is nothing to do: it hashes the chunks it would write, compares them to
# what is stored, and exits without embedding anything. That makes one command
# correct whether the image was built with a key or without one.
set -eu

python -m ingest.pipeline

# Render supplies PORT. The default keeps `docker run -p 8000:8000` working.
exec uvicorn app.main:app --host 0.0.0.0 --port "${PORT:-8000}"
