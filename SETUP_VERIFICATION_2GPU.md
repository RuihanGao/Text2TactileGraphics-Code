# Two-GPU setup verification — 2026-09-08

READY FOR ECCV HARDENING: YES (historical pre-hardening check)

The isolated `deploy/eccv-demo-2gpu` checkout was verified at base commit
`f2975d096d63c02e9edf0fa19493028ae4ff33c6`. The newer deployment task supersedes
the original profiling-only branch in `AGENTS.md`. Current collaborator and
operator instructions are in [ECCV_DEMO.md](ECCV_DEMO.md).

## Reproduced environment

- Linux x86_64; 2× NVIDIA A100-SXM4-80GB, 81920 MiB physical VRAM each.
- Driver 550.127.05; PyTorch 2.11.0+cu128; CUDA runtime 12.8. The driver tool
  reported CUDA 12.4; subsequent hardened full inference is recorded separately.
- Python 3.12.14, project 0.1.0, uv 0.12.10.
- `uv sync --frozen --offline`: exit 0, 154 packages checked. `pyproject.toml`
  and `uv.lock` were unchanged.
- The migrated venv initially had a missing interpreter target and stale editable
  paths. An isolated Python 3.12.14 interpreter was restored under `.venv/python-base`,
  small Hatchling/editables build dependencies were cached, and the editable package
  and console-script paths were corrected to this checkout. The fallback environment
  was preserved; no application dependency versions were changed.

## Caches and credentials

The persistent caches were present and nonempty: Hugging Face about 4.5 GB,
custom checkpoints about 3.7 GB, DiffSynth about 92 GB, approximately 100 GB total.
The cache variables are `HF_HOME`, `TEXT2TACTILEGRAPHICS_CKPT_DIR` and
`DIFFSYNTH_MODEL_BASE_PATH`; the current launcher supplies defaults under
`/workspace/model_cache`.

Required custom files were present:

- `edit-2511-4steps-v1.0-noalpha.safetensors`
- `lightning-4step-v2.0-noalpha.safetensors`
- `texture-step3000.safetensors`

Qwen, MoGe and SAM3 cache directories were found. This setup check established
presence, not complete checkpoint integrity. An inherited HF credential and an
external private Gemini credential file were present; values were never recorded.
No model caches were deleted, relocated or downloaded during verification.

## Tests and isolation

- `.venv/bin/python -m pytest tests/ui/test_quick_trial.py -q`:
  12 passed, 4 deselected, 22.79 seconds.
- `.venv/bin/python -m pytest -q tests`:
  264 passed, 27 deselected, 29.70 seconds. Default tests exclude `slow`.
- Single-GPU launcher smoke on isolated port 8082 returned HTTP 200 with Quick
  Trial. Only that identified temporary process was stopped. This was startup
  validation, not a full inference or residency test.
- Two-GPU and fallback checkouts had separate Git and Python environments.
  No fallback source or upstream remote was modified; nothing was pushed.
- Local tool migration checks passed. No authentication or agent-configuration
  contents are included in this release summary.

No blockers remained for hardening. External access and full dual-GPU inference
were outside this setup check; subsequent evidence is in the
[ECCV readiness report](profiling/results/eccv/ECCV_READINESS_REPORT.md) and
[recovery verification](profiling/results/eccv/RECOVERY_READY_VERIFICATION.md).
The original machine-specific setup report and command appendix remain in ignored
local release evidence. Pod/volume identifiers are intentionally omitted here.
