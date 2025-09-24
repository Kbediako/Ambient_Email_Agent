#!/usr/bin/env bash
set -euo pipefail

export GOOGLE_API_KEY="${GOOGLE_API_KEY:-}"

if [ -z "$GOOGLE_API_KEY" ]; then
  echo "ERROR: GOOGLE_API_KEY not found in env" >&2
  exit 1
fi

echo "✅ GOOGLE_API_KEY detected"
exec "$@"
