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
