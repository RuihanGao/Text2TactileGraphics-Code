# Live Quick Trial validation — 2026-09-08

This report records new testing of the real Quick Trial service. Earlier
supplied-plan and simulated-browser results remain in `QUICK_TRIAL_VALIDATION.md`.
Raw evidence is ignored under `profiling/results/quick-live-20260908/`.

## Environment and scope

Fork `RuihanGao/Text2TactileGraphics-Code`, branch `profile/runpod-a100`, base SHA
`608cd73957db3954cc6245c89c3fc34df473f18e`, with existing uncommitted user work
preserved. No commits or pushes. The advanced app (PID 83181, port 7860) was
preserved. Quick Trial uses a separate process (PID 380620, port 8080).

One NVIDIA A100-SXM4-80GB, driver 570.124.06, 81920 MiB NVML capacity;
Python 3.12.14, PyTorch 2.11.0+cu128, CUDA 12.8, Google GenAI 2.7.0.
Host RAM is 2 TiB; the container limit is 249999998976 bytes, with 27.2 CPU cores
of quota. Shared workspace had about 161 TiB free. The three inherited persistent
cache paths are recorded in `environment.json` and were not changed.

Existing baseline profiling and supplied-plan mesh evidence were inspected before
changes. `/tmp/t2tg-tools/bin/uv sync --frozen --offline` succeeded, checking 154
packages without rebuilding or changing dependencies. The project Python is used
directly by the launcher; `uv` does not need to be on the user's PATH.

## Planner fixes and evidence

1. A real request returned HTTP 400 `INVALID_ARGUMENT`, explicitly rejecting
   `additional_properties` at the root and nested region items of the legacy
   `response_schema`. Changed only the request field to
   `response_json_schema=PromptPlan.model_json_schema()`.
2. That exposed HTTP 404 `NOT_FOUND`: the provider said `gemini-2.5-flash` is no
   longer available to new users and recommended `gemini-3.6-flash`. The default
   text planner now uses that model; the environment override is retained.
3. Both live structured-planner tests passed with `gemini-3.6-flash` (22.45 s).
   They check wings versus avocado texture and separate lamp-base/bark and
   lamp-shade/cloth assignments. This was not a trivial unstructured API probe.

Strict local Pydantic validation remains unchanged, including forbidden extra
fields, region count, bounded text, and ASCII Braille labels. Regression checks
verify the JSON Schema request field and rejection of unexpected provider fields.
No image model, checkpoint, lifecycle implementation, handler, or quality setting
was changed. The request field is also consistent with the installed SDK's JSON
Schema support and [Google's structured-output documentation](https://ai.google.dev/gemini-api/docs/generate-content/structured-output).
Only sanitized provider reasons were emitted; credentials were never printed.

## Launch and access

```bash
cd /workspace/Text2TactileGraphics-Code
./launch_quick_trial.sh
```

The launcher privately sources `/workspace/private/gemini.env`, exports the key,
sets `single_a100_80gb`, and defaults to `0.0.0.0:8080`. Empty/whitespace keys,
missing environments, and occupied ports fail clearly. Empty-key and occupied-port
checks passed without stopping any service. Inherited caches remain intact.
Startup instructions were consolidated around this launcher.

HTTP config checks verified:

- `http://127.0.0.1:8080`: `Text2TactileGraphics · Quick Trial`.
- `http://127.0.0.1:8081`: the same Quick Trial, through the existing nginx proxy.
- `http://127.0.0.1:7860`: `Text-based Tactile Graphics Generation` (advanced app).

External access is **not verified**. The pod-ID-based RunPod URL for port 8081
returned 404; probes for 8080, 7860, and 7861 returned 403 from this environment.
Use the RunPod console's actual HTTP connection/exposure settings for 8081 or
8080. No externally accessible URL is claimed. No nginx configuration was changed.

## Regression checks

- Core suite: **264 passed, 27 deselected**, 28.98 s.
- Focused final Quick Trial suite: **12 passed, 4 deselected**, 18.50 s.
- Live planner: **2 passed, 14 deselected**, 22.45 s.
- Ruff check/format and Bash syntax checks passed for changed code.
- Repository-wide `git diff --check` identifies an existing trailing blank line
  in user-modified `CODEX_INTEGRATE_SINGLE_GPU.md`; it was preserved.

Browser tooling and shared libraries were reused from `/tmp`, outside the project
environment. Real browser testing uses Chromium at 390 × 844 with touch/mobile
emulation, against nginx port 8081 and the production Quick Trial process. No
fixture planner or mocked generation handlers are installed in that service.

## Real browser/GPU runs

All runs used live `GeminiPromptPlanner`, seed 42, 4-step Qwen base and texture,
10-step intra-tile inpainting, full released image resolution, and the existing
final mesh handler with standard Braille. All stages ran sequentially on the
single A100. The downloaded GLBs came from the actual UI Download mesh button.

The first dolphin run reached Done in **308.17 s**, including cold loads and
live planning. A second dolphin in the same server reached Done in **94.67 s**.
The first two-region lamp reached Done in **194.70 s**. These are browser-observed
request latencies, not CUDA-synchronized stage microbenchmarks or controlled
cold/warm averages. Generation, segmentation, texture, tiling, final geometry,
and Done transitions are timestamped in each `events.json`.

The browser harness initially asserted against the wrong DOM element after
successful downloads: Gradio's editable data cells live in `.virtual-row`
containers outside the semantic HTML table. A scroll/wait alone did not fix that
selector. Saved screenshots and DOM confirmed correct plans and visible cells;
the harness now reads `.virtual-row [data-editable=true]`. The earlier assertion
failures are **test harness failures after successful GPU exports**, not passes
of those complete automated browser checks. Their logs remain preserved.

The dolphin plan is `a dolphin with wings`, region `dolphin` with `avocado skin
texture`, Braille `dolphin`. The saved lamp DOM confirms shape/Braille `lamp`,
`base of the lamp` → `tree bark texture`, `lamp shade` → `cloth_bag texture`.
No supplied plan bypassed the live planner during these runs.

The initial mobile layout was visually inspected and had no horizontal overflow,
with customization collapsed and large prompt/Generate controls. Completed
screenshots show the generated object and rendered 3D mesh. A screenshot taken
immediately at the first Done event caught the viewer before rendering finished;
the later screenshots and checks allow loading time.

## Mesh validity

The dolphin exports contain **524754 vertices / 1046172 faces**, 18,851,936 bytes;
the first lamp contains **524732 vertices / 1046485 faces**, 18,855,424 bytes.
Both have finite vertices, consistent winding, and no edges shared by more than
two faces. **Neither is watertight**: dolphin has 3878 boundary edges, lamp 3079.
This reproduces the existing released mesh-closure defect. They load as real
meshes and can be viewed/downloaded, but are not established as print-ready.

No silent hole filling or geometry redesign was applied. The baseline report
already traces the next investigation to `geometry/utils.py:flatten_and_close_plate`,
boundary stitching, and degenerate-face removal; this serving/planner fix does
not change those algorithms.

Credential handling check: the exact saved key was absent from edited project
files and test logs. On this workspace volume, `chmod 700 /workspace/private`
and `chmod 600 /workspace/private/gemini.env` returned success but `stat` still
reported 0777/0666. Thus restrictive Unix modes on that persistent path could
not be verified/enforced here. The key remains outside the repository and is
loaded privately; this report does not claim that the volume honors chmod.

## Final browser result and service state

The corrected automated lamp browser test **passed**, with no page errors. Its
full repeat reached Done in **258.74 s**; the same browser then changed Braille
from `lamp` to `light`, used Apply edits and continue, and downloaded a new mesh
in **47.22 s** to Done. Every intermediate image URL remained unchanged, and
exported vertex coordinates changed. No object generation, region finding, or
tiling stage ran again. The existing orchestration briefly emits cached
“Creating tactile texture” labels when traversing retained regions; those labels
alone are not evidence that texture inference reran. Selective invalidation and
failure/resume handler counts are additionally covered by the focused unit tests.
The rerun mesh has the same boundary-edge defect as the original lamp.

The browser assertions for the lamp cover real plan fields, visible progressive
status, intermediate image HTTP downloads, the final mesh download, mobile
layout, and Braille resume. Dolphin plan/mesh visibility was separately confirmed
from the saved real-browser screenshots after its post-export selector failures;
those failed harness invocations are not retrospectively counted as passing tests.

Approximately 1 Hz NVML telemetry recorded **1155 samples**, 16:07:20–16:26:37 UTC.
Maximum sampled device memory was **64.20 GiB**, final residency **6.92 GiB**;
no CUDA OOM was observed. Utilization reached 100%; its 10.10% overall mean and
99.70 W mean power include CPU mesh work and gaps debugging the browser harness.
These whole-session figures are not per-inference performance averages or a
PyTorch allocator peak measurement. The repeated lamp was slower than the first;
no claim of a stable warm-latency SLA is made.

Quick Trial was left running with the saved server key and `single_a100_80gb`
(PID 380620). The advanced application remains on PID 83181. The dedicated
telemetry process was stopped after validation; neither application was stopped.

Evidence under `profiling/results/quick-live-20260908/`:

- `environment.json`, `environment-sync.log`, `planner-diagnosis.json`.
- `core-tests.log`, `focused-tests.log`, `planner-36-tests.log`, `launcher-tests.json`.
- `first-dolphin/`, `0/`, `first-lamp/`, `1/`: full-run GLBs and stage timestamps.
- `1/`: successful browser report, plan fields, downloaded intermediate WebP
  images, mobile screenshots and DOM. `first-lamp/plan.json` records the prior
  run's actual structured region fields recovered from its saved DOM.
- `braille-rerun/`: second download, stage timestamps, artifact reuse checks,
  changed-geometry check and mesh validity.
- `browser-success.json`, `browser-lamp-recheck.log`: final automated browser pass.
- `browser.log`, `browser-recheck.log`, `browser-lamp.log`: earlier harness failures.
- `mesh-results.json`, per-run `mesh-validity.json`, `gpu.csv`, `gpu-summary.json`.
- `server.log` and the isolated browser/mesh-validation scripts.

Remaining limits: external RunPod access is unverified, Unix credential-file
modes are not enforced by this workspace volume, and exported meshes retain the
released non-watertightness defect. Live Gemini planning and the single-GPU
inference-to-download path are now verified; print-ready geometry is not.
