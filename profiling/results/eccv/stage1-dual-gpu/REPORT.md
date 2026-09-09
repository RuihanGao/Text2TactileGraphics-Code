# Stage 1 — PASS

Hypothesis: the previously exact texture/tiling sharing policy can remain on cuda:1 while base-edit, SAM3 and MoGe remain on cuda:0, removing large Qwen reloads without numerical changes.

Both constructors use Qwen-Image transformer, text encoder, VAE and tokenizer; the differing role resolves to the same explicit cuda:1 in this policy. Base-edit retains its separate Edit-2511 transformer/processor on cuda:0. Installed DiffSynth uses unfused adapter tensor lists, which `clear_lora()` empties; shared execution requires hotload support. The common adapter manager tracks both roles, restores the texture adapters in released order and removes them before tiling. Scheduler state is set by each unchanged pipeline call. Seeds, BF16/CUDA wrapper settings, CFG, resolution and 4/4/10 steps remain unchanged.

Actual new-Pod evidence:

- Dolphin full cold execution: 295.594 s, including Qwen loads 142.26 + 78.18 s. Base, mask, texture, normal PNG and tileable PNG match saved true-80GB references exactly (zero pixel differences).
- Lamp after explicit model warmup: 53.747 s and 53.712 s. Both region masks and all base/texture/normal/tileable images match saved validated references exactly, including return from tiling to texture and a subsequent request. No separate tiling model load.
- Lamp model warmup: 60.187 s with populated caches. Resident allocations: GPU0 59.116 GiB; GPU1 53.807 GiB. Largest lamp reserved peaks: 62.645 / 58.707 GiB.
- Original focused regressions: 17 passed; added lifecycle/advanced UI checks: 33 passed; readiness/Quick Trial checks: 14 passed. No OOM.
- Six live planner schema/semantic checks pass, including textureless input. Median about 2.3 s; no model change justified.
- Mobile browser observed Warming, disabled Generate and no horizontal overflow before readiness. Five real warm browser requests subsequently passed; final measurements are below.

Machine-readable data, synchronized handler/model timings and 1 Hz GPU traces are in the adjacent experiment directories. Warm UI measurements include live planner, all model/geometry work and the Done signal; downloaded files are checked separately. Raw meshes retain the released topology limitations; no geometry changes were made.

## Completed mobile-browser performance

Five sequential real requests, live Gemini, 390×844 Chromium, valid downloadable GLBs, no page errors. Full latency includes click to Done; backend is the sum of disjoint generation/segmentation/texture/normal/tiling/final-mesh handlers, excluding planner, Gradio serialization and browser transport. Results and timestamps are in `browser-performance/summary.json` and `mobile/results.json`.

| Category | p50 s | mean s | min s | max s |
|---|---:|---:|---:|---:|
| full | 45.130 | 45.518 | 44.166 | 47.278 |
| backend | 38.103 | 38.240 | 37.674 | 39.098 |
| planner | 2.183 | 2.465 | 1.835 | 3.528 |
| adapter | 0.988 | 1.050 | 0.875 | 1.390 |
| mesh_handler | 25.448 | 25.419 | 25.096 | 25.741 |
| qwen_load | 0.000 | 0.000 | 0.000 | 0.000 |

No p95 inference from five samples. CPU-dominated mesh handler includes auxiliary GPU estimation; its time is an upper bound for pure CPU geometry, not a separate GPU-free measurement. More detailed geometry spans are enabled for subsequent final-service measurements.

- Are all normal-route large model weights warm before acceptance? **Yes**, both base bundles, SAM3 and MoGe. Small texture LoRAs are reapplied (~1 s); base adapters remain loaded.
- Is texture/tiling sharing exact? **Yes**, all tested reference artifacts including multi-region reuse match exactly.
- GPU0 resident models: base-edit Qwen, SAM3, MoGe.
- GPU1 resident models: shared Qwen-Image for texture/tiling.
- Warm backend latency: **38.240 s mean**.
- Warm full Quick Trial latency: **45.518 s mean**, **45.130 s p50**.
- Planner latency: **2.183 s p50**.
- Peak GPU0/GPU1 allocated VRAM: **61.206/57.149 GiB**; reserved **62.658/58.707 GiB**.
- Remaining per-request model load: **zero large Qwen loads**; texture adapter application is reported separately.
- Numerical equivalence: dolphin and two lamp replicates exact; no intended quality setting changed.
- <60 s achieved: **YES**, five full warm browser requests.

Service startup reached Ready after **103.22 s**, including complete warmup inference. Browser confirmed Warming and disabled Generate first. Browser tooling was installed in `/tmp/eccv-browser` with required Chromium system libraries; application lockfile/dependency versions are unchanged. Raw mesh topology is outside this task.

Final export comparison also passed: the dolphin and both lamp GLBs are byte-identical to the saved single-A100 true-80GB references (`final-mesh-equivalence.json`). This extends the exact image/mask/normal/tile checks through final export.
