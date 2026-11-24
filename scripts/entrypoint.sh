#!/usr/bin/env bash
# Single entry point for the Docker image. Usage: entrypoint.sh <command>
set -euo pipefail

cmd="${1:-serve}"
shift || true

case "$cmd" in
  download) python -m src.data.download_raw ;;
  clean)    python -m src.data.clean ;;
  upload)   python -m src.data.storage upload ;;
  train)    python -m src.train "$@" ;;
  pipeline)                       # raw -> clean -> S3 -> train (pulls from S3)
    python -m src.data.download_raw
    python -m src.data.clean
    python -m src.data.storage upload
    python -m src.train "$@"
    ;;
  serve)    exec uvicorn src.serve.app:app --host 0.0.0.0 --port "${PORT:-8000}" ;;
  test)     exec python -m pytest -q ;;
  *) echo "Unknown command: $cmd (download|clean|upload|train|pipeline|serve|test)"; exit 1 ;;
esac
