#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# API keys live outside version control; export them for Deepgram + Replicate.
if [ -f .env.project1.local ]; then
  set -a
  # shellcheck disable=SC1091
  source .env.project1.local
  set +a
fi

for key in DEEPGRAM_API_KEY REPLICATE_API_TOKEN; do
  if [ -z "${!key:-}" ]; then
    echo "Warning: $key is not set — the app will start but that feature will not work." >&2
  fi
done

exec conda run --no-capture-output -n project1 python -m streamlit run app.py "$@"
