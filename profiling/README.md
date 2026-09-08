# Released pipeline profiling on one A100 80GB

Use the fork's `profile/runpod-a100` branch. No serving or model algorithm changes
are made by this harness. Raw outputs are ignored; keep PROFILE_SUMMARY.md.

Environment: `uv sync --frozen`, then `uv run --frozen pytest -q tests`.
The lockfile is authoritative (its Torch version differs from the README's
historical tested stack). Preserve HF_HOME, TEXT2TACTILEGRAPHICS_CKPT_DIR and
DIFFSYNTH_MODEL_BASE_PATH. Do not put credentials in commands or result files.

The benchmark follows `ui/handlers.py` and the defaults/unit conversions in
`ui/app.py`: Qwen base image, optional base mesh preview, SAM3 dolphin mask,
Qwen avocado-skin texture, MoGe cropped normals, high-pass filtering at 120
with per_channel mode, intra-tile Qwen inpainting at 10 steps, saved segment
(5 mm normal displacement, 3 repeats), and final flattened 4 mm plate with
standard “dolphin” Braille (12 cm plate, 0.3 flat-top ratio, 1 mm padding).
The final mesh uses the application's `create_tactile_graphic`, which performs
base depth estimation, mesh construction, displacement, plate segmentation and
closure, Braille displacement, and GLB export. Human interaction delays are
excluded. Base/texture images remain 1024×1024 and mesh resolution stays 1024.

4-step versus 40-step changes only the released base/texture presets, including
their corresponding LoRA and CFG settings. The tiler stays at its UI default of
10 steps. SDXL is an alternative tiler, not part of the default route; its optional
residency must be labeled separately from the default Qwen route.

Cold means a fresh Python process with populated disk caches. Warm means the
next invocation in that same process. First-time downloads are setup work.
Never report failed attempts as completed warm runs or extrapolate timings for
unexecuted stages. Validate exported GLB watertightness because the released
pipeline skips closure when it cannot segment the plate.

## Commands

Run each command in its own process, sequentially on the GPU:

```bash
uv run --frozen python profiling/profile_pipeline.py --mode 80gb --output profiling/results/natural-80gb
uv run --frozen python profiling/profile_pipeline.py --mode 48gb --output profiling/results/natural-48gb
# If natural residency fails, measure the existing unload APIs separately:
uv run --frozen python profiling/profile_pipeline.py --mode 80gb --lifecycle unload-qwen --output profiling/results/unload-qwen-80gb
uv run --frozen python profiling/profile_pipeline.py --mode 48gb --lifecycle unload-qwen --output profiling/results/unload-qwen-48gb
# Only after a full 4-step path succeeds; retain the same lifecycle:
uv run --frozen python profiling/profile_pipeline.py --mode 48gb --steps 40 --runs 2 --output profiling/results/natural-48gb-40step
```

Each command requests one cold and three warm runs, stops on failure, flushes
partial JSONL records, saves a sanitized traceback, and cleans up loaded models.
`--base-preview` adds the optional Stage 1 base mesh preview. The default takes
the direct route to final export without redundant optional mesh previews.

Stage records have parent IDs: inclusive times overlap; do not sum them. Use
exclusive times to attribute time without double-counting child model loads,
LoRA setup, and device transitions. CUDA is synchronized at both boundaries.
Parent peaks aggregate child peaks and intervening work despite nested resets.
Memory is in GiB. Peak values describe process-wide residency during a span,
not that stage's incremental allocation. A loader's cached_property is inserted
only after its wrapped load returns, so its own resident_models list excludes
the newly loaded name; its after_gib includes that model's memory.

High-level Qwen device transitions are timed separately. On-demand per-layer
preparation/transfers can occur inside inference and remain included there;
the transition total is not a measurement of all CPU/GPU transfer time.
The initial baseline-full-80gb attempt predates transition instrumentation.

`--lifecycle unload` drops generator owners (including cached LoraManagers) and
calls the existing `unload_all_models()` after each model stage. This is an
explicit, conservative lifecycle experiment, not a serving optimization or a
claim that warm weights remain resident. It retains the exact generated input
between stages and includes unloading and subsequent reloading in total time.
Artifacts and watertight validation are included in end-to-end time; UI delays
and Python imports are excluded. Model initialization remains inside cold time.

`--lifecycle unload-qwen` is the narrower residency experiment: release each Qwen
bundle immediately after its stage with `unload_model(name)`, and keep SAM3 and
MoGe cached. Cached generator/LoraManager owners are released first. Run this
same strategy in both VRAM modes for an isolated mode comparison when natural
80GB residency fails. This is harness-only lifecycle management; application
source remains unchanged.

Mesh validity is an artifact field, not an execution abort. The exploratory
natural-48gb run stopped after detecting a nonwatertight GLB; the subsequent
replicate series records this defect while allowing all warm runs to execute.
Each replicate directory saves the exact harness source and SHA256.

## Results and verification

Use new output directory names when repeating experiments; the harness refuses
to overwrite an existing directory. On this pod, uv was bootstrapped at
`/tmp/t2tg-tools/bin/uv`; substitute that executable if uv is not on PATH.

Run `uv run --frozen python profiling/summarize_results.py` after the experiments
to regenerate `aggregate.json` and `STAGE_TABLES.md`. It validates nonnegative
exclusive times and propagation of nested CUDA peaks. `PROFILE_SUMMARY.md`
contains the interpreted results, scope, output-validity defect, and next actions.

The measured matched lifecycle is **unload-qwen**. The more conservative
`unload` option is available in the harness but was not included in the reported
matrix. The 40-step comparison uses one cold and one warm run, versus one cold
and three warm runs for each completed 4-step series. Host cgroup telemetry was
captured separately during this profiling session; GPU telemetry is automatic
for every invocation of the harness.

## True-80GB single-GPU validation

`validate_single_gpu.py` extends this harness by importing its `Recorder`, CUDA
instrumentation, and cleanup functions. It invokes the released UI handlers;
application source and advanced Gradio behavior are unchanged. It requires exactly
one visible GPU and explicitly sets `80gb` mode. The settings remain 1024×1024,
4-step base/texture, seed 42, and the released 512×512 cropped-normal / 10-step
intra-tile / high-pass / displacement / Braille defaults.

The dolphin uses base prompt `a dolphin with wings`, segmentation `dolphin`, and
texture `an avocado skin`. `--example lamp` uses `lamp`, then saves two segments:
`lamp base` / `tree bark` and `lamp shade` / `cloth_bag`, passing the accumulated
list to final export. It does not collapse the regions or substitute sensor maps.

Use a new output directory for every process:

```bash
.venv/bin/python profiling/validate_single_gpu.py --audit \
  --output profiling/results/single-gpu-quality/dolphin-validation
.venv/bin/python profiling/run_single_gpu_references.py
.venv/bin/python profiling/validate_single_gpu.py --audit --reuse \
  --output profiling/results/single-gpu-quality/dolphin-reuse-validation
.venv/bin/python profiling/compare_single_gpu.py \
  --candidate profiling/results/single-gpu-quality/dolphin-reuse-validation \
  --references profiling/results/single-gpu-quality \
  --output profiling/results/single-gpu-quality/equivalence-reuse.json
.venv/bin/python profiling/validate_single_gpu.py --reuse --runs 4 \
  --output profiling/results/single-gpu-quality/dolphin-performance
.venv/bin/python profiling/validate_single_gpu.py --audit --example lamp \
  --output profiling/results/single-gpu-quality/lamp-validation
.venv/bin/python profiling/validate_single_gpu.py --audit --reuse --example lamp \
  --output profiling/results/single-gpu-quality/lamp-reuse-validation
.venv/bin/python profiling/compare_lifecycle_artifacts.py \
  profiling/results/single-gpu-quality/lamp-validation \
  profiling/results/single-gpu-quality/lamp-reuse-validation \
  profiling/results/single-gpu-quality/lamp-lifecycle-artifacts.json
.venv/bin/python profiling/aggregate_single_gpu.py
```

Run GPU commands **serially**. `--audit` captures live parameter dtype/device
inventories, wrapper placements, actual pipeline arguments including defaults,
resolved checkpoint files, LoRA scale and hotload behavior, initial noise hashes,
each denoising update, and scheduler state. Audit hashing synchronizes/copies
small inference tensors and walks modules, so use **no `--audit`** for latency
measurements. Ordinary timing still synchronizes CUDA and measures Qwen inference
separately from model loading. Original PNGs, masks, float normal maps, final GLB,
1 Hz GPU traces, and exact harness copies are saved per process.

`--reuse` is a profiling-only experiment: after base generation it unloads base
Qwen normally. Texture generation loads the released Qwen-Image bundle and its
LoRAs normally. The existing `LoraManager.apply([])` clears the hotloaded adapters;
the clean bundle is then assigned to the tiling cache key. With multiple regions,
it returns to the texture key and the existing generator reapplies the required
LoRAs. It unloads Qwen after the last region and keeps SAM3/MoGe resident across
requests. It asserts hotload support and empty adapter lists before tiling. This
must be validated again if DiffSynth, checkpoints, presets, or adapter semantics
change; it is not an application model-manager refactor.

`quality_index.py DIRECTORY` creates an HTML index and a diagnostic contact-sheet
figure. `render_quality_mesh.py MESH OUTPUT.png` renders the unchanged mesh with
PyVista (requires a working EGL/OSMesa loader). Neither preview enters benchmark
timing. Non-watertight exports remain diagnostics for this deployment task.

## Application lifecycle validation

`validate_application.py` constructs the real Gradio app and invokes its registered
base/texture/normal/tiling/export callbacks, plus the released segmentation and
save-segment handlers. It never installs a lifecycle policy or manually unloads
models. Enable the application policy through its normal environment option:

```bash
TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb uv run --frozen python profiling/validate_application.py --audit --output profiling/results/application-integration/dolphin-equivalence
TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb uv run --frozen python profiling/validate_application.py --runs 3 --output profiling/results/application-integration/dolphin-performance
TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb uv run --frozen python profiling/validate_application.py --example lamp --audit --output profiling/results/application-integration/lamp-equivalence
uv run --frozen python profiling/compare_single_gpu.py --candidate profiling/results/application-integration/dolphin-equivalence --references profiling/results/single-gpu-quality --output profiling/results/application-integration/equivalence.json
```

Use new output directories for reruns. Audit hashing is excluded from the dedicated
cold/two-warm performance series. Times include callback work, synchronized GPU
operations, application-managed unloading, artifact writes and export checks;
imports, browser/network transport and human interaction are excluded.
