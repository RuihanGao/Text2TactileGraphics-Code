# ECCV Quick Trial collaborator guide

## 1. Quick start

```bash
git clone --branch deploy/eccv-demo-2gpu https://github.com/RuihanGao/Text2TactileGraphics-Code.git
cd Text2TactileGraphics-Code
uv sync --frozen
```

Use Linux x86_64 and Python 3.12 (`>=3.12,<3.13`). The lockfile pins
Text2TactileGraphics 0.1.0 and PyTorch 2.11.0+cu128; the validated environment
uses Python 3.12.14, CUDA runtime 12.8 and NVIDIA driver 550.127.05.
The production launcher requires exactly two visible **A100-SXM4-80GB** GPUs.
Do not upgrade dependencies during deployment.

Configure `HF_HOME`, `TEXT2TACTILEGRAPHICS_CKPT_DIR` and
`DIFFSYNTH_MODEL_BASE_PATH` for persistent caches. The launcher's defaults live
under `/workspace/model_cache`; inspect `launch_eccv.sh` for the directory layout.
Download the released custom checkpoints using the research README installation
instructions, then point the checkpoint variable at that directory. Required files:

- `edit-2511-4steps-v1.0-noalpha.safetensors`
- `lightning-4step-v2.0-noalpha.safetensors`
- `texture-step3000.safetensors`

Provide the Qwen, MoGe and SAM3 model caches as well. The verified caches occupied
about 100 GB; a fresh machine needs download time and disk space. Configure
`HF_TOKEN` server-side for model access, including gated weights and their access
approval. Preflight checks cache directories and custom checkpoint presence;
it does not prove every remote model is cached or authenticate providers.

Supply a trusted, private shell credential file outside the checkout containing
an assignment to `GEMINI_API_KEY`. Set `QUICK_TRIAL_KEY_FILE` to its location;
the default is `/workspace/private/gemini.env`. The launcher **requires this file**,
even when a key is inherited in the environment. Never commit its contents or
put credentials in frontend code. Secrets remain server-side; the user's typed
description is sent to the Gemini planner service.

```bash
./launch_eccv.sh --check
./launch_eccv.sh
```

The launcher selects `dual_a100_80gb`, true `80gb` numerical behavior and
`0.0.0.0:8080`. GPU 0 holds base-edit Qwen, SAM3 and MoGe; GPU 1 holds the shared
Qwen-Image texture/tiling bundle with serialized adapter switching. Large model
weights remain resident between requests. Startup performs full inference warmup
and may take several minutes. Wait for HTTP 200 and `ready: true`:

```bash
curl -s -w '\nHTTP %{http_code}\n' http://127.0.0.1:8080/readyz
```

Repeat until Ready; do not use a fixed sleep. Expose internal **HTTP 8080** in
RunPod and open `https://<POD_ID>-8080.proxy.runpod.net`. No nginx mapping is
required. The operator reports laptop and phone access validated for the current
deployment; this packaging task does not repeat external inference tests.

## 2. How the Quick Trial works

```text
Free-form prompt
→ Prompt planner
→ Base image
→ Automatic region segmentation
→ Per-region tactile texture
→ Tiling
→ Final tactile mesh + Braille
```

The one-prompt mobile UI parses shape, textured regions, tactile textures and
Braille label automatically. It displays progress, the final 3D mesh and a GLB
download. The collapsed **Customize intermediate steps** accordion exposes
intermediate images, regions and settings for editing/resume. The separate
advanced research interface in `ui/app.py` is preserved.

## 3. Collaborator UI-styling guide

**SAFE FOR UI / AESTHETIC EDITS — within the sections listed:**

| Exact file | Safe scope |
| --- | --- |
| `src/text2tactilegraphics/ui/public_app.py` | `CSS`, example presentation in `EXAMPLES`/`gr.Examples`, display copy, component labels and layout in `create_demo`; viewer/gallery height and accordion appearance |
| `src/text2tactilegraphics/ui/theme.py` | Theme colors, fonts, spacing and radii for the **advanced** UI; Quick Trial currently uses its own CSS and does not import this theme |

There is no separate Quick Trial CSS/static-asset module. Research image assets
under `src/text2tactilegraphics/assets/` can be generation inputs; do not treat
that directory as disposable branding material.

Safe work includes spacing, typography, colors, buttons, cards, responsive mobile
layout, explanatory copy, example presentation, status styling and viewer sizing.
Keep status text accessible and visible. `public_app.py` also contains backend
wiring: its entire file is **not** safe to rewrite. Keep component identities,
settings values, callback inputs/outputs, escaping, validation, admission tickets,
queue wiring, readiness polling and HTTP routes unchanged. Existing example text
is used by fixtures; coordinate prompt changes and update relevant tests.

## 4. Files collaborators should not change casually

**DO NOT MODIFY FOR VISUAL-ONLY CHANGES**

| Exact file/module | Responsibility |
| --- | --- |
| `src/text2tactilegraphics/config.py` | Lifecycle names, true-80GB settings, explicit GPU placement, checkpoint paths |
| `src/text2tactilegraphics/generation/models.py` | Loading/residency, lifecycle decorator, shared Qwen bundle and adapter ownership |
| `src/text2tactilegraphics/generation/models.py` (`LoraManager`) | Qwen adapter application/switching |
| `src/text2tactilegraphics/generation/base_image_generation.py` | Base model settings and generation |
| `src/text2tactilegraphics/generation/texture_generation.py` | Texture inference and adapters |
| `src/text2tactilegraphics/ui/inference_service.py` | Startup warmup and Ready/Unhealthy lifecycle |
| `src/text2tactilegraphics/ui/admission.py` | Single active workflow, bounded pending queue, ownership and cleanup |
| `src/text2tactilegraphics/ui/quick_pipeline.py` | Orchestration, settings, state and selective invalidation |
| `src/text2tactilegraphics/ui/prompt_planner.py` | Planner prompt/schema/provider semantics |
| `src/text2tactilegraphics/ui/handlers.py`, `src/text2tactilegraphics/ui/state.py` | Generation handlers and model/config ownership |
| `src/text2tactilegraphics/ui/public_app.py` | Callback bodies, `snapshot`, `stream`, `start`, `edit`, `reserve`, `bind`, queue and `create_server` routes |
| `src/text2tactilegraphics/ui/app.py` | Advanced workflow, generation callbacks and runtime GPU controls |
| `src/text2tactilegraphics/geometry/` | Mesh, displacement, plate closure and Braille algorithms |
| `launch_eccv.sh`, `scripts/stop_eccv.py` | Validated launch/stop and process identity |
| `pyproject.toml`, `uv.lock` | Reproducible dependency environment |

These settings and pathways were benchmarked together and checked against
true-80GB reference outputs. Coordinate backend changes explicitly; visual PRs
must preserve checkpoints, model quality, lifecycle, GPU placement and queue policy.

## 5. UI-development workflow without disturbing production

Use a separate clone/environment rather than changing files underneath the live
process. Create a styling branch with `git switch -c ui/my-style-change`. Edit
only the safe sections, then preview using the existing isolated fixture:

```bash
.venv/bin/python profiling/eccv_mock_server.py
```

It binds **127.0.0.1:8083**, supplies a fixed plan, placeholder image and a small
mock mesh, and does not run GPU inference or call Gemini. Open it locally or via
an SSH tunnel. It supports initial Generate/layout review; its lightweight
pipeline has no `resume` method, so it does not validate Customize execution.
Use the focused unit suite for mocked edit/invalidation contracts.

The supplied-plan tests marked `slow` still run real GPU handlers: they are not
lightweight previews. Avoid invoking expensive inference to tweak CSS. Review at
390×844 and desktop sizes, then open a PR from the styling branch rather than
pushing directly to `deploy/eccv-demo-2gpu`.

## 6. Validation commands

From a synced checkout:

```bash
.venv/bin/python -m pytest tests/ui/test_quick_trial.py -q
.venv/bin/python -m pytest -q tests
.venv/bin/ruff check src/text2tactilegraphics/ui/public_app.py src/text2tactilegraphics/ui/theme.py
.venv/bin/ruff format --check src/text2tactilegraphics/ui/public_app.py src/text2tactilegraphics/ui/theme.py
bash -n launch_eccv.sh
git diff --check
./launch_eccv.sh --check
```

Apply Ruff check/format to every Python file changed in your PR. Default pytest
configuration excludes `slow` tests; these commands do not run a new GPU benchmark.
Preflight requires production hardware, caches and credentials; it is safe with
the live server running and does not start another server or load models.

## 7. Production operator notes

- One active workflow uses both GPUs. Default maximum pending is two with a
  180-second projected-wait budget. `TEXT2TACTILEGRAPHICS_MAX_PENDING` and
  `TEXT2TACTILEGRAPHICS_BUSY_WAIT_LIMIT_SECONDS` configure admission;
  `TEXT2TACTILEGRAPHICS_GPU_CONCURRENCY` remains one.
- Warming disables Generate; `/readyz` returns 503 until Ready, or if Unhealthy.
  `/healthz` reports process/readiness/queue status without model loading:
  `curl -fsS http://127.0.0.1:8080/healthz`. Monitor host RSS as well as GPU memory.
- Keep logs, telemetry, screenshots and generated meshes local under ignored
  `profiling/results/` or output directories. `TEXT2TACTILEGRAPHICS_TELEMETRY`
  selects the queue log; `ECCV_MEASUREMENT_DIR` enables optional stage measurement.
  Leave `ECCV_TEST_FAULTS` unset in production.
- For an intentional restart, let the queue drain, then run
  `.venv/bin/python scripts/stop_eccv.py`. It checks command and checkout identity
  before SIGTERM and does not force-kill a draining process. Relaunch and wait for
  Ready. Never use this stop command during CSS work.
- Supported slower fallback, after safely stopping the current service:
  `./launch_eccv.sh --single --check`, then `./launch_eccv.sh --single`.
  This selects GPU 0 and `single_a100_80gb`; `TEXT2TACTILEGRAPHICS_SINGLE_GPU`
  selects another GPU. The same port is used; never start both on 8080.
- Do not casually edit RunPod Pod configuration: container-local tooling can be
  reset. Preserve persistent caches and the known-good single-GPU fallback checkout.
- Raw exports retain the known non-watertightness issue; multi-region segmentation
  can be imperfect. The public URL is Pod-specific unless fronted by a stable
  redirect. Production concurrency intentionally remains one workflow at a time.

Historical measurements and limitations: [ECCV readiness report](profiling/results/eccv/ECCV_READINESS_REPORT.md).
The post-push `ECCV_RELEASE_HANDOFF.md` is a local, ignored receipt with the exact
release SHA and push outcome; collaborator instructions above ship in Git.
