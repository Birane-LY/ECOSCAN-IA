#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")"
gateway="$(ip route show default | awk '$1 == "default" { print $3; exit }')"

if [[ -z "$gateway" ]]; then
  printf 'Unable to determine the WSL host gateway.\n' >&2
  exit 1
fi

export DJANGO_API_URL="http://${gateway}:8000"
export DJANGO_HOST_HEADER="localhost"
export TEXT_MODEL_CHAIN="${TEXT_MODEL_CHAIN:-groq:openai/gpt-oss-120b,groq:qwen/qwen3.6-27b}"

exec venv/bin/uvicorn app.main:app --host 127.0.0.1 --port 8001
