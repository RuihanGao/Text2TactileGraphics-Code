# Quick Trial validation — 2026-09-08

The sections below record the **earlier supplied-plan validation**, not the
current credential state. New live-server verification is recorded in
[the live validation report](QUICK_TRIAL_LIVE_VALIDATION.md).

Implementation is on the repository-instructed `profile/runpod-a100` branch,
based on `608cd73957db3954cc6245c89c3fc34df473f18e`. The working tree initially
used `demo/quick-trial`; no changes were made there. No commits were pushed.
The advanced `src/text2tactilegraphics/ui/app.py`, generation handlers,
checkpoints, and model lifecycle implementation are unchanged.

## Environment

- One NVIDIA A100-SXM4-80GB, driver 570.124.06, 81920 MiB reported by NVML.
- Python 3.12.14; PyTorch 2.11.0+cu128; CUDA runtime 12.8; Gradio 6.24.0.
- Host RAM: 2.0 TiB total (host-level reading, not the pod memory quota).
- Workspace filesystem: 162 TiB available (shared filesystem reading).
- `uv sync --frozen --offline` succeeded before implementation and after the
  dependency declaration. Pydantic was already locked/installed at 2.13.4;
  adding it as a direct dependency did not change any package versions.
- Existing profiling evidence was read before implementation; its baseline
  and full inference results are in `profiling/results/PROFILE_SUMMARY.md`.
- Baseline core tests: 252 passed, 23 slow tests deselected.

## Real GPU execution

Both tests used `TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb` and the
released advanced handlers. There were no mocked image, segmentation, texture,
tiling, or geometry operations. **The planner was a fixture supplying the
expected structured plans**, because this server has no Gemini key.

| Input | Shape / regions | Full pipeline seconds | Vertices | Faces | Export |
| --- | --- | ---: | ---: | ---: | --- |
| Dolphin with wings and avocado skin texture | `a dolphin with wings`; `dolphin` → `avocado skin texture`; Braille `dolphin` | 276.36 | 524754 | 1046172 | GLB succeeded |
| Lamp with bark base and cloth-bag shade | `lamp`; `base of the lamp` → `tree bark texture`, `lamp shade` → `cloth_bag texture`; Braille `lamp` | 129.00 | 524732 | 1046485 | GLB succeeded |

These are single validation executions, sequentially in one process, not a
controlled performance comparison. The initial run included cold model loads.
Both use 4-step base/texture and 10-step intra-tile inpainting, matching the
advanced interface presets. Two GPU integration tests passed in 421.21 seconds
including imports and validation. Both exports have nonzero vertices/faces;
**neither is watertight**, reproducing the existing documented geometry defect.
A successful viewer/download path does not establish print-ready mesh quality.

Raw evidence is gitignored under `output-quick-trial/`:

- `environment.json`, `gpu-tests.log`, `core-tests.log`
- `0/result.json`, `1/result.json`: plans, settings policy, latency, mesh counts
- `0/events.json`, `1/events.json`: progressive stage timestamps
- Each numbered folder: `base.png`, per-region masks/textures/tiled PNGs, `final.glb`
- `browser.json`, `mobile-start.png`, `mobile-done.png`: browser verification

## UI and dependency tests

Final core suite: **264 passed, 27 slow tests deselected** (26.30 seconds).
The two live-planner checks explicitly skipped because credentials are absent.
Ruff lint, format checks, and `git diff --check` passed.

Twelve focused tests cover both sample plans, provider request/schema contracts,
malformed output, missing credentials, multi-region handler execution, selective
invalidation, uploaded-image validation, stage-failure recovery, textureless
objects, session isolation, region insertion/reordering, stale editor clearing,
the collapsed single-column layout, and the shared callback queue.

A Chromium browser at 390 × 844 with mobile/touch emulation verified:

- The default screen and closed customization accordion; no horizontal overflow.
- One-button generation, progressive output updates, final viewer, and successful
  transfer of the mesh through the download button.
- Editing Braille and continuing called only the final mesh handler again;
  base generation, segmentation, texture, geometry, and tiling counts stayed unchanged.
- No JavaScript page errors. The final progress component uses a polite ARIA
  live region. Screenshots were inspected.

The browser used simulated inference and a fixture planner to isolate UI behavior;
the separate GPU tests above exercised real inference. Browser tooling and shared
libraries were installed only under `/tmp`, outside the project environment.
The local test server was not published.

## Remaining validation limits

1. **Live planner:** `GEMINI_API_KEY`/`GENAI_API_KEY` are absent. Two live-planner
   tests skip explicitly. Mocked structured responses establish schema/provider
   wiring, not the language model's actual interpretation. Configure the key on
   the server and run `pytest tests/ui/test_quick_trial.py -m slow -k live_prompt`
   to validate both prompts against the live provider. No key belongs in the UI.
2. **Dual GPU:** the unchanged backend rejects `dual_a100_80gb` with
   `ValueError: Unknown model lifecycle`. Frontend code has no lifecycle branches
   or device assignments, but actual dual-GPU startup requires backend support.
3. **Geometry:** both real GLBs are non-watertight. The existing defect is outside
   this UI task; no repair, quality reduction, or checkpoint substitution was made.
