# A100 profiling summary

Measured on 2026-09-07, fork `RuihanGao/Text2TactileGraphics-Code`, branch `profile/runpod-a100`, source SHA `2f55fdc5c0a3b046b08cd055ddd8bd8d71805623`.

## Outcome and required answers

The released Qwen route executes through segmentation, texture generation, tiling, displacement, Braille, and GLB export on one A100 80GB. Natural **80gb mode runs out of VRAM while loading texture Qwen**. The existing unload API makes the full 80gb route executable. **Exported meshes fail watertightness validation**, so successful execution is not a claim of print-ready output.

- **Primary bottleneck:** Qwen model residency and movement. Natural 48gb mode repeatedly moves weights during generation. In the fastest tested lifecycle, repeated Qwen initialization/loading is the largest measured category (37.68 s per warm run).
- **Secondary bottleneck:** CPU mesh assembly and displacement; final mesh handler averages 26.55 s in the fastest tested configuration.
- **Warm end-to-end latency:** 81.06 s with 80gb + targeted Qwen unloading (three warm runs, range 78.96–84.57 s). Natural forced-48gb: 194.71 s.
- **80GB vs 48GB speedup:** 2.29× with the **same targeted-unload lifecycle**. Natural-cache speedup is undefined because natural 80gb fails before completing a cold run; no invented natural-80gb warm numbers.
- **Peak VRAM:** successful 80gb targeted-unload series: 61.66 GiB allocated, 63.24 GiB reserved. Natural 80gb OOM attempt: 78.20 GiB allocated. See the memory table for NVML samples and forced-48gb peaks.
- **Can all necessary models coexist:** not GPU-resident in natural 80gb mode. All default-route bundles can remain cached under forced-48gb CPU offloading, at substantial host-memory cost. Targeted unloading retains SAM3/MoGe and one Qwen bundle at a time. Optional SDXL and click-tracker bundles are not required by this route and were not loaded.
- **Time attributable to model loading/offloading:** the fastest warm configuration spends 37.68 s loading models, 1.88 s on LoRA setup, 2.02 s explicitly unloading Qwen, and 0.29 s in high-level device transitions. Natural forced-48gb spends 83.86 s in high-level transitions per warm run. High-level device-transition time is measured separately; per-layer preparation/transfers inside inference are not fully separable by these hooks.
- **Recommended public-demo GPU configuration:** the tested single A100 80GB with targeted Qwen unloading. Preserve generous persistent model-cache storage and host RAM; the pod has a 250 GB decimal (~233 GiB) RAM limit. Lower host-memory allocations were not benchmarked. Forced-48gb is not a physical 48GB fit guarantee.
- **Recommended public-demo inference configuration:** released 4-step Qwen base/texture presets, 1024×1024 images, default 10-step intra-tile inpainting, unchanged geometry and Braille defaults. Prewarm with a full run; provide queued/asynchronous progress because even the fastest measured warm path takes over a minute. No demo was implemented.
- **Recommended next optimization:** after addressing mesh validity, adopt the measured Qwen lifecycle and investigate reusing the texture/tiling Qwen bundle with the correct LoRA switching to avoid repeated loads. Also profile caching base mesh construction for repeated texture/Braille edits. These are proposals, not implemented serving or model refactors.

## Environment and reproduction

`uv sync --frozen` reproduced the lockfile. Python 3.12.14, PyTorch 2.11.0+cu128, CUDA runtime 12.8; one NVIDIA A100-SXM4-80GB, driver 570.124.06. NVIDIA-SMI advertises 81920 MiB; CUDA reports 79.25 GiB usable total. The README's historical tested stack (Python 3.12.13 / Torch 2.9.1 / driver 580.x) differs from the current lockfile. The lockfile pins Torch 2.11.0+cu128 and allows Python 3.12.x; dependencies and lockfile were not rewritten to match the historical README.

Linux x86_64; 256 visible CPUs but a **27.2-core cgroup quota**. Host RAM is about 2 TiB, but pod RAM is limited to 249999998976 bytes. Initial shared-volume free space was 160.6 TiB (not a claim about a private pod quota). CPU throttling and container-memory samples are retained in `host-trace.jsonl`; GPU telemetry is per experiment. Initial shell Python was 3.11.10 and uv was absent; uv was bootstrapped at `/tmp/t2tg-tools/bin/uv`.

Caches inherited and verified:

- `HF_HOME=/workspace/model_cache/huggingface`
- `TEXT2TACTILEGRAPHICS_CKPT_DIR=/workspace/model_cache/text2tactilegraphics/ckpt`
- `DIFFSYNTH_MODEL_BASE_PATH=/workspace/model_cache/diffsynth`

All three released custom safetensors were downloaded and header/checksum-validated (`custom-checkpoints.json`). Hugging Face access, including gated SAM3, worked. MoGe uses `model.pt`; an initial probe for an unused `config.json` returned 404 and was corrected. No Gemini key was required or used.

**Core tests: 246 passed, 23 slow tests deselected.** A subsequent focused slow geometry regression also passed (1 test, 28.24 s); it validates a vertex snapshot and does not assert watertightness. The unmodified `BaseImageGenerator.generate` successfully generated the representative image before instrumentation. FlashAttention 2.8.3 installed with the release's CUDA-build-skip setting lacks `flash_attn_2_cuda`; the tested Qwen path uses its released PyTorch SDPA fallback successfully. No attention backend replacement was made.

## Scope and timing definitions

Prompt `a dolphin with wings`; segment `dolphin`; texture `an avocado skin`; seed 42. Base/texture steps 4, tiling steps 10 (UI default, not reduced to 4); normal-mode center crop and high-pass settings remain at UI defaults. Segment displacement 5 mm, normal direction, three repeats. Final plate thickness 4 mm; standard `dolphin` Braille, 12 cm plate, 0.3 flat-top ratio, 1 mm bottom padding.

The harness invokes real Gradio handlers and instruments their actual core calls. It follows the direct route to final export, excluding optional base/textured mesh previews and display-only preview rendering. Base depth is computed inside the final mesh handler; texture normals use MoGe earlier. Human delays and Python imports are excluded. Model initialization, LoRA setup, artifact writes, export validation, and in-run unloads are included in end-to-end latency. Post-series cleanup is outside it.

Cold = fresh Python process, populated persistent disk caches. OS page-cache residency was not forcibly standardized. Cold load variation is material: base Qwen loaded in 35 s in the initial attempt but 124 s in the later lifecycle cold run; warm lifecycle loads were 12–17 s. Reported cold numbers therefore describe these runs, not a guaranteed startup time.

Each GPU span synchronizes before/after, resets peak stats, and records allocated/reserved/peak/free/total memory in GiB. Nested parent peaks aggregate child/intervening peaks. Inclusive spans overlap; use exclusive timings for non-overlapping attribution. `STAGE_TABLES.md` and `aggregate.json` retain detailed stage and per-run values. GPU trace sampling is approximately 1 Hz and can miss sub-second peaks.

## End-to-end experiments

| Configuration | Cold s | Warm 1 s | Warm 2 s | Warm 3 s | Warm mean s |
|---|---:|---:|---:|---:|---:|
| Natural 80gb (OOM) | 160.58 | not available | not available | not available | not available |
| Natural forced-48gb | 285.21 | 207.30 | 187.74 | 189.09 | 194.71 |
| Targeted unload, 80gb | 306.22 | 84.57 | 79.65 | 78.96 | 81.06 |
| Targeted unload, forced-48gb | 206.52 | 185.44 | 183.38 | 187.47 | 185.43 |
| Natural forced-48gb, 40 steps | 346.60 | 332.30 | not available | not available | 332.30 |

Both natural-80gb attempts stopped at a cold-run OOM; the table uses the final frozen-harness confirmation: there are no valid natural-80gb warm runs. The exploratory natural-48gb run executed/exported in 224.71 s but deliberately aborted on watertight validation; it is excluded from replicate averages. The subsequent harness records geometry validity separately and completes the full repeat series.

## Handler-stage timing

Cold and warm-mean inclusive seconds. Handler spans include child model loading; do not add the model-load table to these values.

| Handler | Natural 48 cold | Natural 48 warm | Unload 80 cold | Unload 80 warm | Unload 48 cold | Unload 48 warm |
|---|---:|---:|---:|---:|---:|---:|
| handler.base_image | 53.91 | 58.20 | 131.67 | 20.12 | 55.63 | 51.08 |
| handler.segment | 14.65 | 0.53 | 17.08 | 0.46 | 4.75 | 0.51 |
| handler.texture | 115.80 | 53.81 | 94.55 | 15.36 | 49.19 | 46.84 |
| handler.texture_geometry | 11.30 | 0.13 | 11.99 | 0.11 | 8.38 | 0.12 |
| handler.tiling | 54.10 | 53.52 | 15.10 | 15.11 | 47.04 | 46.58 |
| handler.final_mesh | 30.47 | 27.07 | 28.78 | 26.55 | 26.42 | 27.12 |

The final mesh span includes base depth, mesh construction/conversion, texture displacement, a second SAM3 call for plate segmentation, closure, Braille, and GLB export. Detailed substage timings are in `STAGE_TABLES.md`.

## Core geometry and export timings

Inclusive seconds for the actual core functions; these are already contained in the handler totals above.

| Core function | Natural 48 cold | Natural 48 warm mean | Unload 80 cold | Unload 80 warm mean |
|---|---:|---:|---:|---:|
| depth | 0.119 | 0.097 | 0.085 | 0.088 |
| mesh_construction | 10.751 | 10.100 | 10.515 | 9.561 |
| mesh_conversion | 0.322 | 0.288 | 0.310 | 0.286 |
| texture_integration | 11.024 | 9.285 | 9.938 | 9.276 |
| plate_closure | 5.600 | 4.812 | 5.363 | 4.811 |
| braille | 2.132 | 1.958 | 2.043 | 1.982 |
| export | 0.051 | 0.052 | 0.050 | 0.046 |

## Model loads, residency, and device movement

| Category | Natural 48 cold | Natural 48 warm | Unload 80 cold | Unload 80 warm | Unload 48 cold | Unload 48 warm |
|---|---:|---:|---:|---:|---:|---:|
| Model initialization/load | 33.53 | 0.00 | 254.43 | 37.68 | 18.93 | 6.10 |
| LoRA setup | 5.76 | 0.00 | 2.16 | 1.88 | 2.11 | 2.32 |
| High-level device transitions | 101.45 | 83.86 | 0.34 | 0.29 | 92.65 | 85.99 |
| Explicit Qwen unloads | 0.00 | 0.00 | 2.12 | 2.02 | 11.12 | 11.82 |

| Model load | Unload 80 cold s | Unload 80 warm mean s | Natural 48 cold s |
|---|---:|---:|---:|
| qwen_base_edit | 123.72 | 13.89 | 3.09 |
| sam3_text | 16.43 | 0.00 | 14.08 |
| qwen_texture | 90.88 | 11.80 | 3.71 |
| moge2 | 11.49 | 0.00 | 10.93 |
| qwen_tiling | 11.90 | 11.98 | 1.71 |

SAM3/MoGe have no warm loads in the targeted policy because they remain cached. Qwen owners, including cached LoraManagers, are released before calling the existing `unload_model(name)`; otherwise references can prevent reclamation. Raw records show memory before and after each load/unload. In the 80gb cold lifecycle run, unloading texture Qwen reduced allocated memory from about 58.4 to 3.3 GiB in under a second. Natural 80gb had already retained about 57.9 GiB after base generation and SAM3 segmentation, then failed to allocate another 18 MiB during texture Qwen initialization.

### Measured residency in the targeted 80gb cold run

These are cumulative allocated GiB immediately after each loader returns. Qwen bundles are explicitly unloaded between stages, so rows must not be summed. LoRA setup follows Qwen initialization and can increase residency.

| Just loaded | Allocated GiB |
|---|---:|
| load.qwen_base_edit | 53.792 |
| load.sam3_text | 3.160 |
| load.qwen_texture | 57.061 |
| load.moge2 | 4.499 |
| load.qwen_tiling | 58.294 |
| After full export, Qwen bundles unloaded | 4.499 |

## Peak memory and GPU telemetry

| Configuration | Allocated peak GiB | Reserved peak GiB | Sampled NVML used peak GiB | Peak process RSS GiB | Mean GPU utilization % | Mean power W |
|---|---:|---:|---:|---:|---:|---:|
| Natural 48 | 45.66 | 50.81 | 51.77 | 181.61 | 13.40 | 99.07 |
| Unload 80 | 61.66 | 63.24 | 61.57 | 5.20 | 17.12 | 108.62 |
| Unload 48 | 44.37 | 49.34 | 49.90 | 64.43 | 28.54 | 112.79 |
| Natural 48 / 40 steps | 44.05 | 47.84 | 48.40 | 179.09 | 53.00 | 208.80 |

Utilization means cover initialization, GPU inference, CPU geometry, and cleanup. In the 4-step experiments, low averages and substantial measured device-transition/load time show that the workload is not sustained GPU denoising. Memory-utilization percentages and used/free VRAM are preserved in each CSV. Process RSS excludes the shared filesystem page cache; low RSS in the 80gb unload policy does not imply its fast reload timings can be reproduced with only that much host RAM.

## Host-resource observations

| Policy | Covered seconds / experiment seconds | Throttled CPU periods % | Peak cgroup memory GiB |
|---|---:|---:|---:|
| Natural 48 | 165 / 869 | 0.85 | 232.83 |
| Unload 80 | 548 / 549 | 1.94 | 107.21 |
| Unload 48 | 763 / 763 | 1.43 | 166.48 |
| Natural 48 / 40 steps | 678 / 679 | 1.12 | 232.83 |

The natural-48gb host trace began late in that series, so its coverage is partial. The later series have near-complete coverage. Cgroup memory includes file cache and other processes in the pod, unlike process RSS. Natural caching approached the pod memory limit; no host OOM was observed. The 80gb targeted policy used much less host memory including file cache, but a lower-RAM pod was not tested. Throttled-period percentages are not wall-time fractions.

## 4-step versus 40-step

The comparison ran only after full 4-step execution succeeded. It uses natural forced-48gb mode in separate fresh processes, with identical other settings and the tiler fixed at 10 steps. The 40-step experiment includes one cold and one warm run; its smaller sample is not a latency distribution estimate. Warm 40-step latency is 332.30 s versus 194.71 s for 4-step (1.71×). These are the released presets: 40 steps also removes the distillation LoRA and uses CFG 4 rather than CFG 1. This is not a pure linear step-count scaling test or a perceptual quality study. Four steps remains the latency-oriented public-demo default.

## Mesh defect, limits, and next actions

The first forced-48gb GLB has 524752 vertices, 1046135 faces, **3933 boundary edges**, no nonmanifold edges, and consistent winding (`mesh-validity.json`). Plate closure actually ran; this was not merely an empty plate-segmentation fallback. All 14 planned full-run exports (12 at 4 steps and two at 40 steps) fail watertightness, as does the exploratory export. Base/texture/tile PNG hashes are identical across all four natural-48gb runs and all four 80gb targeted-unload runs (`cross-mode-image-checksums.json`), so the measured lifecycle change did not alter these generated images.

Do not label the output print-ready. Next geometry action: reproduce the saved GLB's boundary edges and trace `geometry/utils.py:flatten_and_close_plate`, boundary stitching, and degenerate-face removal on that generated input. This profiling task does not alter the released geometry algorithm or silently repair exports.

Reproduce the fastest tested full route on this pod with a new output directory:

```bash
/tmp/t2tg-tools/bin/uv run --frozen python profiling/profile_pipeline.py --mode 80gb --lifecycle unload-qwen --output profiling/results/recheck-unload-80gb
```

Reproduce the output-validity issue without GPU inference:

```bash
.venv/bin/python -c 'import trimesh; m=trimesh.load_mesh("profiling/results/natural-48gb/cold-final.glb"); print(m.is_watertight)'
```

The final immutable-harness natural-80gb confirmation reproduced the same texture-load OOM (`natural-80gb-confirmation/`), after 160.58 s of partial end-to-end execution. Natural-80gb residency OOM is the precise blocker to a natural-cache cold/three-warm comparison. The matched targeted-unload 80gb/48gb series measures the existing lifecycle remedy instead. No quantization, checkpoint replacement, image-resolution reduction, model-architecture change, serving rewrite, or public demo implementation was performed. No commits were pushed. All code additions are under `profiling/`; dependency files and application source remain unchanged.

## Evidence inventory

- `environment.json`, `environment-locks.sha256`, `uv-sync.log`, `core-tests.log`
- `checkpoint-access.json`, `custom-checkpoints.json`, setup/download logs
- `base-baseline.log`, `baseline-base.png` (unmodified released baseline)
- Per-experiment `settings.json`, `stages.jsonl`, `gpu.csv`, images and final GLBs; sanitized error traces where applicable
- Replicate `harness.py` and SHA256 snapshots; `host-trace.jsonl`
- `aggregate.json`, `STAGE_TABLES.md`, `mesh-validity.json`, image-checksum records
- `verification.json`, `geometry-regression.log`, `secret-scan.json`, and final-check logs

Raw artifacts are gitignored and retained locally. This summary is the non-ignored deliverable. Run counts, immutable harness hashes, exclusive-time sums, nested peaks, and non-overlapping benchmark windows were verified. A text-artifact scan found no inherited credential values. Benchmark numbers describe serial requests on this pod; concurrency, lower-RAM pods, other GPUs, and final-mesh correctness fixes have not been benchmarked.
