# Application integration on one A100 80GB

Validated 2026-09-07 on one NVIDIA A100-SXM4-80GB, branch `profile/runpod-a100`,
base SHA `9fb317a5b4fbfafd84c70acb415584d34f7f4796`, with the uncommitted integration
recorded by [application source hashes](application-integration/dolphin-performance/source-sha256.json).
The initial checkout was `deploy/single-a100` at that same SHA; switched to the
branch explicitly required by `AGENTS.md`, preserving all existing changes.
No commits were pushed and no upstream remote was modified.

## Required answers

- **Does the actual Gradio/core application use true 80gb numerical mode?** Yes.
  `TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb` selects sequential Qwen
  unloading while preserving the released BF16/CUDA wrapper settings, checkpoints,
  adapters, scheduler, seeds, image sizes, CFG and 4/4/10-step defaults. Incompatible
  48gb settings raise an error; there is no fallback.
- **Does it complete on one A100?** Yes: dolphin equivalence, a fresh-process
  cold/two-warm dolphin series, and the complete two-region lamp example all export
  successfully through application callbacks/handlers. No OOM occurred.
- **Peak VRAM?** 61.665 GiB allocated /
  63.236 GiB reserved across all runs.
  CUDA reports 79.252 GiB usable. Dolphin performance sampled NVML peak:
  61.567 GiB; 1 Hz samples can miss brief allocator peaks.
- **Cold application latency?** 110.046 s in the dedicated performance process,
  with populated disk caches. An earlier audited fresh-process run took 258.532 s,
  including 176.368 s of Qwen loading; disk/page-cache state is material and was
  not standardized. The 110 s result is not a guaranteed first-launch time.
- **Warm application latency?** 91.684 s mean; 92.190 s and
  91.177 s for the two measured warm runs.
- **Are representative outputs consistent with validation?** Yes, exact file
  equality for all eight dolphin artifacts and all fourteen lamp artifacts,
  including masks, float normal arrays, generated images and final GLBs. Every
  dolphin performance replicate also matches the previous targeted-unload outputs.
- **What files were modified?** Listed below. The advanced UI and all existing
  generation/geometry algorithms remain available.
- **What remains before a public demo?** Resolve the latency-policy choice,
  implement the separately scoped simplified UX, and add deployment supervision,
  request limits, progress/error handling and artifact retention. The known lamp
  selection limitation and non-watertight raw exports remain; print readiness needs
  separate geometry validation/repair. No simplified UI or geometry fix was made.

## Lifecycle integration

The core `qwen_stage` wrapper covers base-edit generation, texture generation,
and both Qwen tiling generators. It holds a per-manager lock during a Qwen call,
releases cached generator LoRA managers afterward, and invokes the existing
`ModelManager.unload_model`. This applies on success and failure; finished
exception frames are cleared to prevent their local references retaining weights.
Weak references verify reclamation of the pipeline, transformer, text encoder and
VAE, and unreclaimed references produce an explicit error. Debug logs record
allocated/reserved GiB and `retained=[]` after every successful unload.

No CPU offload is added. No extra `empty_cache()` call is added: reclamation uses
the existing unload API's garbage collection/cache release, as in the validated
harness. SAM3 and MoGe stay cached. After full workflows, allocated residency is
about **4.499 GiB**, with only these auxiliary bundles cached. Three Qwen unloads
occur per dolphin run and five per two-region lamp run.

The application runs this policy without importing anything from `profiling/`.
The research default remains `cached`. In single-A100 mode all nine GPU-consuming
Gradio callbacks share a concurrency group with limit one, covering base and
texture generation, both segmentation routes, geometry, tiling and mesh exports.
Debug GPU/VRAM controls reject incompatible changes; other controls remain intact.

## Numerical and representative validation

`validate_application.py` constructs `app.create_demo()` and invokes its actual
registered base/texture/normal/tiling/export callbacks, including the UI's unit
conversions and fixed tiling settings. Segmentation and saved-region assembly use
the released handlers with that same application's `AppState`. The validator
contains **no model unload, generator eviction, cache aliasing or lifecycle patch**.
Timing and optional tensor-audit hooks observe the application policy.

The dolphin's base, texture and tiling outputs exactly match the independently
saved true-80gb references. Actual inference arguments, component dtypes/devices,
adapter inventories, scheduler state, initial noise and all recorded denoising
steps match. [Numerical comparison](application-integration/equivalence.json) and
[full dolphin artifact/runtime comparison](application-integration/dolphin-artifacts.json).

Examples use seed 42 and the released separate controls:

- `a dolphin with wings with an avocado skin texture.` → base `a dolphin with wings`,
  segment `dolphin`, texture `an avocado skin`.
  [Artifact index](application-integration/dolphin-performance/index.html),
  [base](application-integration/dolphin-performance/cold-base.png),
  [texture](application-integration/dolphin-performance/cold-region1-texture.png),
  [GLB](application-integration/dolphin-performance/cold-final.glb).
- `lamp, the base of the lamp has a tree bark texture, and the lamp shade has a cloth_bag texture.`
  → base `lamp`, regions `lamp base` / `lamp shade`, generated textures `tree bark` /
  `cloth_bag`. Both saved regions are applied to the final export. All five Qwen
  calls and fourteen artifacts match the previously validated targeted-unload
  lamp workflow. [Comparison](application-integration/lamp-artifacts.json),
  [artifact index](application-integration/lamp-equivalence/index.html),
  [combined GLB](application-integration/lamp-equivalence/cold-final.glb).
  The audited lamp request took 153.018 s; it is not a warm performance estimate.

The lamp-base mask still selects the lower pedestal rather than the entire support.
No segmentation improvement was attempted. Both raw GLBs remain non-watertight,
byte-identical to the prior exports; this is an existing diagnostic limitation.
Optional SDXL, Gemini, click refinement, different step presets and larger tiling
settings were not GPU-benchmarked in this integration task.

## Application performance

| Run | E2E s | Qwen loading s | Explicit unload s | Peak allocated GiB | Peak reserved GiB |
|---|---:|---:|---:|---:|---:|
| cold | 110.046 | 45.288 | 3.067 | 60.427 | 61.992 |
| warm1 | 92.190 | 43.989 | 2.883 | 61.665 | 63.236 |
| warm2 | 91.177 | 43.550 | 2.183 | 61.665 | 63.236 |

Timings synchronize CUDA around measured spans and aggregate nested allocator
peaks after resetting each span's peak statistics. E2E includes callback work,
model loading, inference, application unloading, artifact saving and export checks.
Imports/UI construction, browser/network transport, human delays and optional
intermediate mesh previews are excluded. The performance series disables tensor
hashing and parameter inventories. [Raw stages](application-integration/dolphin-performance/stages.jsonl),
[GPU trace](application-integration/dolphin-performance/gpu.csv),
[aggregate and per-artifact stability](application-integration/aggregate.json).

**The ~75 s target is not reached by this explicit separate-unload policy.** The
previous 75.301 s mean used texture/tiling bundle reuse and only two Qwen loads;
this task explicitly requires unloading texture Qwen and separately loading tiling
Qwen. The measured 91.684 s warm mean is 16.383 s (21.8%) above that
reuse target. Warm Qwen loading averages
43.769 s, versus 28.928 s for reuse.
The earlier separate-unload profiling mean was 81.06 s; this application series is
10.624 s slower, with current Qwen loading also slower than the prior 37.68 s.
These are comparisons across runs with uncontrolled filesystem/page-cache state,
not a controlled measurement of pure UI overhead. Reuse is the logical next
lifecycle optimization if the explicit unload-after-texture requirement is relaxed;
it was not silently introduced here.

## Environment and regression checks

Frozen `uv sync --frozen` verified the released environment without changing
`pyproject.toml` or `uv.lock`. Python 3.12.14, PyTorch 2.11.0+cu128, CUDA runtime
12.8, driver 570.124.06. Persistent HF/custom-checkpoint/DiffSynth caches were
inherited. CPU/RAM/disk and cache locations are in
[environment.json](application-integration/environment.json).

Core tests passed before integration (246 tests; 23 slow deselected). After
integration the full suite passes **252 tests, 23 slow deselected**, including
cleanup/retry, external-reference detection, serialized Qwen calls, unchanged
cached policy, UI queue membership and incompatible-setting rejection.
[Final test log](application-integration/core-tests-final.log).
Changed Python files pass Ruff and `git diff --check`.

## Files changed in this task

- `src/text2tactilegraphics/config.py`: explicit lifecycle configuration/validation.
- `src/text2tactilegraphics/generation/models.py`: scoped Qwen lifecycle and verified cleanup.
- `src/text2tactilegraphics/generation/base_image_generation.py`: base Qwen scope.
- `src/text2tactilegraphics/generation/texture_generation.py`: texture Qwen scope.
- `src/text2tactilegraphics/generation/tileable_patch_generation.py`: both Qwen tiling scopes.
- `src/text2tactilegraphics/ui/app.py`: shared inference queue, guarded debug settings, logging.
- `tests/generation/test_qwen_lifecycle.py`, `tests/ui/test_app.py`: lifecycle/UI regressions.
- `README.md`: normal application launch/configuration instructions.
- `profiling/validate_application.py`: actual application callback validation driver.
- `profiling/README.md`, `profiling/.gitignore`: reproduction instructions/report exception;
  pre-existing edits to these files were preserved.
- `profiling/results/APPLICATION_INTEGRATION_REPORT.md`: this report.

Other pre-existing untracked profiling scripts/reports and task files were preserved.
Generated images, meshes, logs, telemetry and comparison JSON remain gitignored.

## Gradio launch

The existing application is launched using the project's documented entry point:

```bash
TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb \
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 \
uv run --frozen gradio src/text2tactilegraphics/ui/app.py
```

Launch process and HTTP readiness evidence are saved in
[server.json](application-integration/server.json); logs are in
[gradio.log](application-integration/gradio.log). The normal advanced UI is served
on port 7860. No profiling harness is involved in the server process.
