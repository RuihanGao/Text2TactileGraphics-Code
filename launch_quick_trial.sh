#!/usr/bin/env bash
# Credentials are loaded only into this process, never copied into the repository.
set +x
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
secret_file="${QUICK_TRIAL_KEY_FILE:-/workspace/private/gemini.env}"
if [[ ! -r "$secret_file" ]]; then
  echo 'Quick Trial: server credential file is missing or unreadable.' >&2
  exit 1
fi
if ! source "$secret_file" >/dev/null 2>&1; then
  echo 'Quick Trial: could not load server credential file.' >&2
  exit 1
fi
set +x
if [[ -z "${GEMINI_API_KEY:-}" || "$GEMINI_API_KEY" == *[[:space:]]* ]]; then
  echo 'Quick Trial: GEMINI_API_KEY is empty or contains whitespace; update the private credential file.' >&2
  exit 1
fi
export GEMINI_API_KEY
export TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb
export GRADIO_SERVER_NAME="${GRADIO_SERVER_NAME:-0.0.0.0}"
export GRADIO_SERVER_PORT="${GRADIO_SERVER_PORT:-8080}"
if [[ ! -x .venv/bin/python ]]; then
  echo 'Quick Trial: project .venv/bin/python is missing; reproduce uv.lock first.' >&2
  exit 1
fi
# Detect occupied ports without terminating any existing service.
.venv/bin/python - <<'PY'
import os
import socket
import sys

try:
    with socket.socket() as probe:
        probe.bind((os.environ['GRADIO_SERVER_NAME'], int(os.environ['GRADIO_SERVER_PORT'])))
except (OSError, ValueError):
    sys.exit('Quick Trial: cannot bind the configured host/port. Choose a free GRADIO_SERVER_PORT.')
print('Quick Trial: server key loaded; lifecycle single_a100_80gb.', flush=True)
PY
exec .venv/bin/python -u -m text2tactilegraphics.ui.public_app
