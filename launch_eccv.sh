#!/usr/bin/env bash
# Deterministic poster launch. Secrets stay in the process environment.
set +x
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
single=0
check_only=0
for arg in "$@"; do
  case "$arg" in
    --single) single=1 ;;
    --check) check_only=1 ;;
    *) echo 'Usage: ./launch_eccv.sh [--single] [--check]' >&2; exit 2 ;;
  esac
done
secret_file="${QUICK_TRIAL_KEY_FILE:-/workspace/private/gemini.env}"
if [[ ! -r "$secret_file" ]]; then
  echo 'ECCV: private planner credential file is missing.' >&2
  exit 1
fi
if ! source "$secret_file" >/dev/null 2>&1; then
  echo 'ECCV: private planner credential file could not be loaded.' >&2
  exit 1
fi
set +x
if [[ -z "${GEMINI_API_KEY:-}" || "$GEMINI_API_KEY" == *[[:space:]]* ]]; then
  echo 'ECCV: planner credential is empty or malformed.' >&2
  exit 1
fi
export GEMINI_API_KEY
export HF_HOME="${HF_HOME:-/workspace/model_cache/huggingface}"
export TEXT2TACTILEGRAPHICS_CKPT_DIR="${TEXT2TACTILEGRAPHICS_CKPT_DIR:-/workspace/model_cache/text2tactilegraphics/ckpt}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-/workspace/model_cache/diffsynth}"
export TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=dual_a100_80gb
if [[ "$single" == 1 ]]; then
  export CUDA_VISIBLE_DEVICES="${TEXT2TACTILEGRAPHICS_SINGLE_GPU:-0}"
  export TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb
fi
export GRADIO_SERVER_NAME=0.0.0.0
export GRADIO_SERVER_PORT=8080
export TEXT2TACTILEGRAPHICS_POSTER_MODE=1
export TEXT2TACTILEGRAPHICS_MAX_PENDING="${TEXT2TACTILEGRAPHICS_MAX_PENDING:-2}"
export TEXT2TACTILEGRAPHICS_GPU_CONCURRENCY=1
export TEXT2TACTILEGRAPHICS_BUSY_WAIT_LIMIT_SECONDS="${TEXT2TACTILEGRAPHICS_BUSY_WAIT_LIMIT_SECONDS:-180}"
export TEXT2TACTILEGRAPHICS_TELEMETRY="${TEXT2TACTILEGRAPHICS_TELEMETRY:-/tmp/text2tactilegraphics-events.jsonl}"
export ECCV_CHECK_ONLY="$check_only"
.venv/bin/python - <<'PY'
import os
import socket
from pathlib import Path

import torch
from text2tactilegraphics.config import Config
from text2tactilegraphics.ui.admission import Admission

required = 1 if os.environ['TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE'] == 'single_a100_80gb' else 2
if torch.cuda.device_count() != required or any(
    torch.cuda.get_device_name(i) != 'NVIDIA A100-SXM4-80GB'
    or torch.cuda.get_device_properties(i).total_memory < 75 * 1024**3
    for i in range(required)
):
    raise SystemExit(f'ECCV: requires exactly {required} visible A100-SXM4-80GB GPU(s).')
config = Config(vram_mode='80gb')
Admission.from_environment()
for key in ('HF_HOME', 'DIFFSYNTH_MODEL_BASE_PATH'):
    if not Path(os.environ[key]).is_dir():
        raise SystemExit(f'ECCV: required cache {key} is missing.')
if not all(Path(p).is_file() and Path(p).stat().st_size for p in config.ckpt_paths.values()):
    raise SystemExit('ECCV: required custom checkpoints are missing or empty.')
if os.environ['ECCV_CHECK_ONLY'] != '1':
    with socket.socket() as probe:
        try:
            probe.bind(('0.0.0.0', 8080))
        except OSError:
            raise SystemExit('ECCV: port 8080 is occupied. Stop only the identified demo process first.')
print(f'ECCV: configuration validated; lifecycle={config.model_lifecycle}; true 80gb; GPUs={required}.', flush=True)
print('ECCV: startup will report Warming; accept visitors only after /readyz returns Ready.', flush=True)
PY
if [[ "$check_only" == 1 ]]; then exit 0; fi
printf '%s\n' "$$" > "${TEXT2TACTILEGRAPHICS_PID_FILE:-/tmp/text2tactilegraphics-eccv.pid}"
# Optional read-only instrumentation for the measured deployment. Production
# behavior and model ownership remain in the application in both entry points.
if [[ -n "${ECCV_MEASUREMENT_DIR:-}" ]]; then
  exec .venv/bin/python -u profiling/eccv_serve.py
fi
exec .venv/bin/python -u -m text2tactilegraphics.ui.public_app
