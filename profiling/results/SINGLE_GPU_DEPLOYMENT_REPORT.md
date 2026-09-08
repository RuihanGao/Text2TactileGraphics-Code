# Single-A100 deployment validation

Measured 2026-09-07 on **one NVIDIA A100-SXM4-80GB**. **One A100 is sufficient for the tested true-80GB outputs and complete released workflows.** Targeted Qwen unloading preserves isolated-80GB outputs exactly. A separately validated texture/tiling reuse experiment also preserves outputs and reduces the dolphin warm mean from the previous **81.06 s to 75.30 s**.

No forced `48gb` execution, additional GPU, checkpoint substitution, quantization, precision reduction, resolution reduction, or application/UI change was used. The lifecycle implementation remains under `profiling/`; deploying it requires adopting that lifecycle in the application. The unmodified natural-cache path still has the previously established OOM constraint.

## Environment and scope

- Fork `RuihanGao/Text2TactileGraphics-Code`, branch `profile/runpod-a100`, source SHA `9fb317a5b4fbfafd84c70acb415584d34f7f4796`.
- Application source, `pyproject.toml`, and `uv.lock` are unchanged from the prior measured application SHA `2f55fdc5c0a3b046b08cd055ddd8bd8d71805623`. Each process saves exact validation/recorder source copies; this report includes uncommitted profiling additions.
- Python **3.12.14**, PyTorch **2.11.0+cu128**, CUDA runtime **12.8**, driver **570.124.06**. Frozen `uv` environment reused and verified; **246 core tests passed, 23 slow tests deselected**. Profiling lint and compilation passed.
- One visible CUDA device, enforced by the harness. NVIDIA-SMI reports **81920 MiB** total; CUDA reports **79.2517 GiB** usable. Every model role maps to `cuda:0`.
- Pod RAM limit **249999998976 bytes (~232.83 GiB)**; CPU quota **27.2 cores**. Host reports about 2 TiB RAM; shared-volume free space approximately 161 TiB. These are not minimum deployment requirements or private storage quotas.
- Existing persistent HF, custom-checkpoint, and DiffSynth caches were reused. No weights were changed. A byte-level check verified **19 Qwen safetensors files, 91.79 GiB**, against their Hugging Face LFS SHA256 identities; all three custom-checkpoint SHA256 values match the prior baseline.

Machine and checkpoint evidence: [environment](single-gpu-quality/environment.json), [checkpoint identities/revisions](single-gpu-quality/checkpoint-identities.json), [verified Qwen bytes](single-gpu-quality/verified-qwen-weights.json), [core tests](single-gpu-quality/core-tests.log).

## Numerical equivalence

### Isolated true-80GB references

Each reference ran in a **fresh Python process**, with only its required Qwen bundle loaded. The tiling reference reads the exact saved dolphin normal PNG from the targeted-unload run and applies the same released high-pass handler. There is no independently regenerated input between the tiling pair.

| Output | Isolated 80GB PNG SHA256 | Targeted-unload 80GB PNG SHA256 | Exact match | Max / mean absolute pixel difference |
|---|---|---|---|---|
| Base image | `e3e54a6c24d686d4aecbc4b58982bbc188ed50d0379c340b59fba5565f55105d` | `e3e54a6c24d686d4aecbc4b58982bbc188ed50d0379c340b59fba5565f55105d` | Yes | 0 / 0 |
| Avocado texture | `f2928dc9b2a0b9ddd2c026aa9f1d5151098d09cd72a3d4d1ab84e542c3fcae08` | `f2928dc9b2a0b9ddd2c026aa9f1d5151098d09cd72a3d4d1ab84e542c3fcae08` | Yes | 0 / 0 |
| Tileable texture | `bdfa76a2c8dcf0a9be9437c7d4965d30b914689f90b100744f7169340ceb5f8c` | `bdfa76a2c8dcf0a9be9437c7d4965d30b914689f90b100744f7169340ceb5f8c` | Yes | 0 / 0 |

The **reuse candidate has these same three hashes**. For every pair, actual call arguments including defaults, component classes, parameter dtypes/devices, adapter inventories, resolved checkpoint/loader configurations, and final scheduler state match. Texture and tiling construction configurations are equal; their logical role names differ intentionally.

**No divergence was observed:** initial noise tensors, every recorded denoising-step output, and final image pixels match exactly. There is consequently no first divergent operation to explain. These are tests of isolated **80GB versus 80GB lifecycles**, not evidence derived from forced-48GB output matches.

Evidence: [targeted comparison](single-gpu-quality/equivalence-targeted.json), [reuse comparison](single-gpu-quality/equivalence-reuse.json), [consistency/residency checks](single-gpu-quality/verification.json). Original reference images are in [reference-base](single-gpu-quality/reference-base/reference.png), [reference-texture](single-gpu-quality/reference-texture/reference.png), and [reference-tiling](single-gpu-quality/reference-tiling/reference.png). Each process directory contains `runtime.jsonl`, `settings.json`, stage timings, and GPU telemetry.

### Actual runtime configuration

All Qwen transformer, text-encoder, and VAE parameters were **BF16 on `cuda:0`**. The inventories cover all parameters, not one sampled parameter: transformer 20,430,401,088; text encoder 8,292,166,656; VAE 126,892,531. Every observed Qwen offload/onload/preparing/computation wrapper setting was BF16/CUDA. No CPU-offload numerical configuration was used.

| Stage/model | Checkpoint identity | Adapters and actual scale | Scheduler / steps / CFG | Seed and noise device |
|---|---|---|---|---|
| `qwen_base_edit` | Qwen-Image-Edit-2511 transformer; Qwen-Image text encoder/VAE/default tokenizer; Qwen-Image-Edit processor | `edit-2511-4steps-v1.0-noalpha.safetensors`, scale 1 | FlowMatchScheduler, Qwen-Image preset; 4 steps; CFG 1 | 42; CPU noise generator; inference CUDA |
| `qwen_texture` | Qwen-Image transformer/text encoder/VAE/tokenizer | `texture-step3000.safetensors`, then `lightning-4step-v2.0-noalpha.safetensors`; scale 1 each | Same scheduler class/preset; 4 steps; CFG 1 | 42; CUDA:0 noise generator |
| `qwen_tiling` | Same Qwen-Image base components/tokenizer as texture | **None**, verified empty adapter lists | Same scheduler class/preset; 10 steps; CFG 4; denoising strength 0.9 | 42; CPU noise generator; inference CUDA |
| SAM3 text | `facebook/sam3` | None; FP32 parameters on CUDA:0 | No diffusion scheduler/CFG/steps; confidence and mask thresholds 0.5 | No stochastic generation seed parameter in released handler |
| MoGe v2 | `Ruicheng/moge-2-vitl-normal` | None; FP32 parameters on CUDA:0 | No diffusion scheduler/CFG/steps; released `infer` default FP16 autocast unchanged | No stochastic generation seed parameter in released handler |

Revisions: Qwen-Image `75e0b4be04f60ec59a75f475837eced720f823b6`; Edit-2511 `6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9`; Edit processor `ac7f9318f633fc4b5778c59367c8128225f1e3de`; custom checkpoint repository `2032de134aec59557c1618c2d7a4c552dead1cbb`; SAM3 `3c879f39826c281e95690f02c7821c4de09afae7`; MoGe `cb0e8bbd6b1e243589717c78e750b1ba4c093acf`.

Full prompts, conditioning-image pixel hashes, sizes, blur settings, negative prompts, LoRA paths/scales, scheduler arrays, and live wrapper inventories are recorded in `runtime.jsonl`. The base uses the released marble-plate conditioning, style prefix/suffix, automatic edit-image resizing, and `zero_cond_t=True`. Base and texture generation are 1024×1024. MoGe normal estimation runs on the full texture and uses the released 512×512 center crop; tiling stays 512×512 with high-pass threshold 120 / `per_channel`, 64-pixel seam mask, blur size 4 / sigma 1. This crop is the existing UI default, not a benchmark resolution reduction.

### RNG validation

Released generators explicitly pass seed **42** into every Qwen call. Installed DiffSynth constructs a new `torch.Generator(rand_device).manual_seed(seed)` inside `generate_noise`, then passes that generator to `torch.randn`. Thus model construction cannot advance the generation noise stream. Base and tiling use the released CPU noise default; texture explicitly uses CUDA:0. These differences were preserved. No global reseeding or deterministic-backend change was introduced.

The runtime hook captured seed/noise-device arguments and hashed the resulting initial noise and each scheduler update. Equality also held after unloading/reloading and after clearing/reapplying adapters. Exact installed RNG/LoRA/wrapper source is preserved in [runtime-source](single-gpu-quality/runtime-source/).

## Representative quality examples

Both semantic requests were expressed through the **existing separate base, segmentation, texture, tiling, saved-segment, and final-export handlers**. No new natural-language parser or substitute interface was introduced.

### 1. `a dolphin with wings with an avocado skin texture.`

Base prompt `a dolphin with wings`; segmentation `dolphin`; texture `an avocado skin`; seed 42 and the runtime settings above. The complete workflow succeeded with both targeted unloading and reuse. All **8 primary artifacts**, including the float normal array, mask, and final GLB, are byte-identical between lifecycles. The performance cold/three-warm series also reproduces every artifact exactly.

Recommended-lifecycle artifacts: [visual index](single-gpu-quality/dolphin-performance/index.html), [base image](single-gpu-quality/dolphin-performance/cold-base.png), [segmentation](single-gpu-quality/dolphin-performance/cold-region1-segmentation.png), [texture](single-gpu-quality/dolphin-performance/cold-region1-texture.png), [normal](single-gpu-quality/dolphin-performance/cold-region1-normal.png), [tileable patch](single-gpu-quality/dolphin-performance/cold-region1-patch.png), [final GLB](single-gpu-quality/dolphin-performance/cold-final.glb), [mesh preview](single-gpu-quality/dolphin-performance/mesh-front.png).

Visual inspection shows a recognizable winged dolphin with crisp relief edges; the mask includes its body, wings, and tail. The generated texture has recognizable pebbled avocado skin, and the final relief carries the corresponding rounded surface pattern. The default 5 mm displacement makes that pattern pronounced; small residual relief details are visible. This is a successful representative reproduction, not a claim that the default mesh is artistically or physically optimal.

### 2. `lamp, the base of the lamp has a tree bark texture, and the lamp shade has a cloth_bag texture.`

Base prompt `lamp`. Region 1 uses segmentation `lamp base` and generated texture `tree bark`. Region 2 uses segmentation `lamp shade` and generated texture **`cloth_bag`**. The released `save_segment` appends both masks/patches to one list, and final export applies both enabled segments. The bundled sensor-based `cloth_bag.png` was **not** substituted for generated texture.

The complete two-region workflow succeeded with both lifecycles. All **14 primary artifacts** and the runtime/intermediate outputs of **five Qwen calls** match exactly. This additionally tests the clean pipeline returning from tiling to texture and reapplying both texture adapters. Evidence: [lamp lifecycle comparison](single-gpu-quality/lamp-lifecycle-artifacts.json).

Recommended-lifecycle artifacts: [visual index with both regions](single-gpu-quality/lamp-reuse-validation/index.html), [base image](single-gpu-quality/lamp-reuse-validation/cold-base.png), [base-region mask overlay](single-gpu-quality/lamp-reuse-validation/cold-region1-segmentation.png), [bark texture](single-gpu-quality/lamp-reuse-validation/cold-region1-texture.png), [bark normal](single-gpu-quality/lamp-reuse-validation/cold-region1-normal.png), [bark tile](single-gpu-quality/lamp-reuse-validation/cold-region1-patch.png), [shade mask overlay](single-gpu-quality/lamp-reuse-validation/cold-region2-segmentation.png), [cloth texture](single-gpu-quality/lamp-reuse-validation/cold-region2-texture.png), [cloth normal](single-gpu-quality/lamp-reuse-validation/cold-region2-normal.png), [cloth tile](single-gpu-quality/lamp-reuse-validation/cold-region2-patch.png), [combined GLB](single-gpu-quality/lamp-reuse-validation/cold-final.glb), [mesh preview](single-gpu-quality/lamp-reuse-validation/mesh-front.png).

Visual inspection shows a recognizable lamp, vertically structured bark texture, and woven cloth texture, with distinct surface treatments in the combined relief. **The released `lamp base` text selection covers the lower pedestal, not the entire urn-shaped support below the shade.** Its mask contains 25,723 pixels; the shade contains 134,602 pixels; overlap is zero. Both requested texture assignments are represented, with this explicit region-selection limitation. Broader base coverage would require refinement through the existing selection UI; changing GPU count does not address that semantic selection. The cloth normal map has subtle weave that the default displacement makes more conspicuous in the mesh.

This is a limited visual sanity check, not a perceptual study, print validation, or proof for every prompt/preset. The independent isolated-reference tests use the dolphin inputs; the lamp tests compare the two complete true-80GB lifecycles.

## Safest tested lifecycle improvement

Inspection established that texture and tiling construct the same `QwenImagePipeline`, `QwenImageDiT`, `QwenImageTextEncoder`, VAE, and tokenizer from identical base files and loader dtype/device configurations. The base-edit transformer is a **different checkpoint** and was not shared.

In the installed DiffSynth implementation, the active VRAM-managed LoRA path stores separate `lora_A_weights` / `lora_B_weights`. It applies their contributions during forward execution; it does not fuse and later subtract rounded BF16 deltas from base weights. `LoraManager.apply([])` invokes the existing `clear_lora()` and clears these lists. The experiment asserts hotload support and absence of adapters before tiling.

Mutable state considered:

- Adapter tensors and `LoraManager.loaded_paths`: explicitly cleared before tiling; both required adapters reapplied in released order for the next texture.
- Scheduler timesteps/sigmas: overwritten by the released `set_timesteps` call on every generation; final state and every denoising update matched references.
- Wrapper onload/offload/preparation state: all configured BF16/CUDA; existing high-level transitions remain active.
- Conditioning/latents: per-call inputs stay local to the released pipeline. No prompt, normal-map, RNG, or latent cache was introduced.

Recommended sequence: load/generate/unload base-edit Qwen; retain SAM3/MoGe as needed; load texture Qwen, apply texture adapters, generate texture, clear adapters, run tiling on that same clean base. For another region, reapply the correct texture adapters and repeat. **Unload the shared Qwen bundle after the last region, before the next base-edit load.** Generator/LoraManager owners are released along with model-manager cache entries. Use the existing unload API. About **4.50 GiB allocated** remains for SAM3/MoGe after full export.

Natural caching of all three Qwen bundles remains unsupported on this A100: the prior baseline OOM while loading texture Qwen is retained as evidence and was not needlessly rerun. The tested lifecycle requires only one Qwen bundle plus auxiliary models at a time. No architectural transformer sharing across incompatible checkpoints was implemented.

## Performance of the recommended lifecycle

Dedicated **fresh-process cold + three warm** dolphin runs, after numerical equivalence passed, with audit tensor hashing disabled. Same seed, prompts, resolution, steps, LoRAs, geometry, Braille, and export path as the validated workflow. GPU timings synchronize before/after and reset/aggregate peak allocator statistics. Model loading, inference, explicit unloading, geometry, and artifact/export work are separated.

Cold means first request in a fresh process with populated persistent model caches; imports and post-series cleanup are outside E2E. OS page caches were not flushed. This is not first-ever model download or guaranteed process-launch latency. Validation-only runs include extra tensor hashing/module inventory and are excluded from performance comparisons. The lamp reuse validation took 110.19 s with auditing; no three-warm lamp performance distribution was measured.

| Measurement | Cold | Warm 1 | Warm 2 | Warm 3 | Warm mean |
|---|---:|---:|---:|---:|---:|
| E2E seconds | 97.365 | 75.507 | 75.446 | 74.950 | **75.301** |
| Qwen loading seconds | 33.463 | 29.247 | 29.130 | 28.407 | **28.928** |
| Qwen inference seconds | 12.054 | 11.013 | 11.003 | 11.335 | **11.117** |
| CPU geometry/export seconds | 29.862 | 27.767 | 28.101 | 27.913 | **27.927** |
| Allocated peak GiB | 60.601 | 61.662 | 61.662 | 61.662 | — |
| Reserved peak GiB | 61.992 | 63.223 | 63.223 | 63.223 | — |

Warm **min 74.950 s; max 75.507 s**. Best observed request is 74.950 s; the reproducible three-run mean is 75.301 s.

| Loading/lifecycle category | Cold seconds | Warm mean seconds |
|---|---:|---:|
| Base-edit Qwen load | 18.258 | 15.027 |
| Texture/shared Qwen load | 15.205 | 13.901 |
| Separate tiling Qwen load | **0** | **0** |
| SAM3 load | 3.913 | 0 |
| MoGe load | 8.006 | 0 |
| All model initialization/load | 45.382 | 28.928 |
| LoRA setup/clearing | 2.079 | 2.496 |
| Explicit Qwen unloading | 1.442 | 1.471 |
| High-level device transitions | 0.591 | 0.406 |

Auxiliary inference/core estimation contributes **1.137 s warm mean**, in addition to **11.117 s Qwen inference**: approximately **12.253 s actual generation/estimation**. Qwen inference includes high-level device transitions, so do not add that transition row again. Geometry uses exclusive times for construction, conversion, displacement, closure, Braille, and export. The remaining E2E time includes handler work, image/array writes, validation, and uncategorized orchestration. Machine-readable exclusive categories sum to E2E; inclusive parents are not double-counted.

Peak **sampled NVML VRAM: 63.776 GiB**; peak process RSS **6.624 GiB**. The performance series captured 321 approximately 1 Hz NVIDIA-SMI samples: mean GPU utilization **26.16%**, memory-controller utilization **5.37%**, mean power **128.83 W**. Sampled peaks can miss short allocator peaks. Low average GPU utilization is consistent with substantial loading and CPU geometry time, not continuous denoising. RSS excludes filesystem page cache; no smaller-host-RAM configuration was tested.

Compared with the previous **81.06 s** targeted-unload warm baseline, the new mean is **5.76 s / 7.10% faster (1.076×)**. Qwen loading drops from **37.68 s to 28.93 s**, an **8.75 s** reduction. End-to-end improvement is smaller because measured geometry, adapter, and other costs differ; the new harness also preserves more intermediate evidence. The previous 306.22 s cold result is not a controlled cold-speedup comparison because OS cache/I/O conditions differ. No forced-48GB numbers enter the quality conclusion.

**Remaining bottlenecks are nearly tied: Qwen loading (28.93 s) and CPU geometry/export (27.93 s).** Keeping all Qwen bundles resident on additional GPUs could potentially reduce reload latency; no multi-GPU speedup was measured or assumed.

Raw evidence: [aggregate](single-gpu-quality/aggregate.json), [performance stages](single-gpu-quality/dolphin-performance/stages.jsonl), [1 Hz GPU trace](single-gpu-quality/dolphin-performance/gpu.csv), [all stability checks](single-gpu-quality/verification.json).

## Geometry/export diagnostics

Both workflows export nonempty GLBs with the expected segment counts and Braille. Dolphin: 524,752 vertices / 1,046,135 faces. Lamp: 524,732 vertices / 1,046,462 faces. Raw meshes are **not watertight**, consistent with the existing baseline; this is diagnostic information and **not a blocker for this deployment decision**. No topology repair was attempted. Identical final GLBs across lifecycles show that reuse did not introduce this condition.

Diagnostic mesh rendering needed an EGL loader absent from the container. It was extracted under `/tmp/t2tg-render-libs`, without changing the Python lockfile or exported geometry. Rendering and full checkpoint hashing were outside the dedicated performance series.

## Deployment conclusion — required answers

**Can one A100 80GB reproduce true-80GB-quality outputs?** Yes, for the tested released presets and examples, with every Qwen stage using BF16/CUDA true-80GB execution.

**Does targeted unloading change numerical outputs?** No change was observed: all three isolated-reference PNGs, initial noise, and denoising updates match exactly. The separate adapter-reuse optimization also passes; both complete example workflows produce identical artifacts between lifecycles.

**Recommended single-GPU lifecycle:** True `80gb`; unload base-edit Qwen after generation; keep SAM3/MoGe cached; reuse the compatible Qwen-Image base for texture/tiling by clearing and correctly reapplying the released hotloaded adapters; unload that bundle before the next base-edit request. The original targeted-unload lifecycle remains the validated fallback.

**Warm single-GPU latency:** **75.30 s mean**, **74.95–75.51 s**, for the one-region dolphin at released 4/4/10 steps. This is not a measured multi-region warm latency.

**Peak VRAM:** **61.66 GiB allocated, 63.22 GiB reserved, 63.78 GiB sampled NVML** across the recommended lifecycle tests.

**Remaining Qwen loading overhead:** **28.93 s warm mean** (15.03 s base-edit + 13.90 s texture/shared base), plus 2.50 s adapter setup/clearing and 1.47 s explicit unloading.

**Can the current repository be deployed on one A100 80GB without using 48gb numerical mode?** Yes, when it adopts the tested lifecycle. The current natural-cache application path still needs that lifecycle integration; this task deliberately leaves application/UI code unchanged.

**Is more GPU memory required for quality/correctness?** No additional GPU memory is required for the demonstrated outputs and lifecycle. Natural simultaneous residency of all three Qwen bundles does require more memory, but it is unnecessary for correct sequential execution. Remaining semantic-selection and raw-mesh limitations are present in identical true-80GB outputs and are not GPU-capacity defects.

**Would renting additional GPUs provide capacity, quality, latency, or only concurrency benefits?** They could provide **latency benefits from persistent model residency and concurrency benefits**. They are not required for quality or operational capacity of this sequential pipeline. Their larger aggregate capacity would enable a different caching policy; any speedup remains unmeasured. The benefits would not be limited to concurrency, but no rental is justified by an observed quality or single-request capacity failure here.

**Recommended next experiment before renting multiple GPUs:** Measure whole-bundle host-memory staging or cached deserialization of the remaining base-edit and shared Qwen models, restoring all BF16 weights to CUDA before inference and repeating these exact equivalence checks. Keep true-80GB inference semantics. Compare cold/three-warm latency and host RAM against this 75.30 s baseline; the measured 28.93 s reload budget makes this a concrete single-GPU target.

Reproduction commands and safety assertions are in [profiling README](../README.md). Raw evidence is gitignored and retained under `profiling/results/single-gpu-quality/`; this report is not ignored. No commits were pushed, no upstream remote was modified, and no public demo was built.
