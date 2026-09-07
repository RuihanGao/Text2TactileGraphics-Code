# Text2TactileGraphics RunPod Profiling Instructions

## Mission

Profile the released Text2TactileGraphics pipeline on a **single NVIDIA A100 80GB RunPod GPU** and determine what is required for a fast public web demo.

Upstream repository:

* `alex4727/Text2TactileGraphics`

Development fork:

* `RuihanGao/Text2TactileGraphics-Code`

Working branch:

* `profile/runpod-a100`

## Working rules

1. Work only in this fork and this branch.
2. Never modify or push to the `upstream` remote.
3. Never commit or print secrets, Hugging Face tokens, API keys, SSH keys, `.env` files, or Codex authentication files.
4. Do not push commits unless explicitly asked.
5. Baseline the released pipeline before optimizing it.
6. Preserve the existing advanced Gradio interface.
7. Keep profiling code isolated under `profiling/`.
8. Do not quantize, retrain, replace checkpoints, reduce image quality, or alter model architecture merely to make the benchmark pass.
9. Make reasonable implementation decisions autonomously. Do not stop for routine reversible decisions.
10. Surface genuine blockers such as missing credentials, CUDA OOM, corrupt checkpoints, or a dependency that cannot be reproduced.

## Expected machine

* Linux x86_64
* 1× NVIDIA A100 80GB
* Python 3.12
* project managed using `uv`

Use the persistent caches inherited from the shell:

```bash
HF_HOME=/workspace/model_cache/huggingface
TEXT2TACTILEGRAPHICS_CKPT_DIR=/workspace/model_cache/text2tactilegraphics/ckpt
DIFFSYNTH_MODEL_BASE_PATH=/workspace/model_cache/diffsynth
```

Do not display secret environment-variable values.

## Baseline first

Before modifying profiling or inference code:

1. Inspect `README.md`, `pyproject.toml`, `uv.lock`, and the relevant source.
2. Record:

   * Git SHA and branch
   * GPU model
   * driver version
   * GPU VRAM
   * CPU RAM
   * disk space
   * Python version
   * PyTorch version
   * CUDA runtime version
3. Reproduce the released environment.
4. Run the core tests.
5. Establish that the existing inference pipeline can execute a representative example.

Do not refactor before these steps.

## Important repository behavior

The project already has Qwen VRAM modes:

* `80gb`: Qwen loading/offloading remains on CUDA.
* `48gb`: Qwen onload/offload uses CPU while computation occurs on CUDA.

A required experiment is therefore to run both modes on the **same A100 80GB**, using identical input and settings.

The `ModelManager` lazily caches multiple large models, including:

* `qwen_base_edit`
* `qwen_texture`
* `qwen_tiling`
* MoGe
* SAM3
* SDXL tiling

It provides:

```python
unload_model(...)
unload_all_models()
```

Measure model residency before proposing changes to it.

## Profiling implementation

Add:

```text
profiling/
├── README.md
├── profile_pipeline.py
└── results/
```

Raw generated profiling outputs should be gitignored.

### Accurate CUDA timing

CUDA kernels are asynchronous. Synchronize before and after timed GPU operations:

```python
torch.cuda.synchronize()
start = time.perf_counter()

# measured operation

torch.cuda.synchronize()
elapsed = time.perf_counter() - start
```

Before each measured stage:

```python
torch.cuda.reset_peak_memory_stats()
```

Record at least:

```python
torch.cuda.memory_allocated()
torch.cuda.memory_reserved()
torch.cuda.max_memory_allocated()
torch.cuda.max_memory_reserved()
torch.cuda.mem_get_info()
```

Report GiB.

Separate **model initialization/load time** from **inference time**.

## Required stages

Profile, where present in the current released path:

1. Qwen base model initialization
2. base-image generation
3. MoGe initialization
4. base depth/normal/relief estimation
5. SAM3 initialization
6. text-based segmentation
7. texture Qwen initialization
8. tactile texture generation
9. tiling-model initialization
10. tileable texture generation
11. displacement / texture integration
12. mesh construction
13. Braille generation/integration
14. final watertight mesh/export
15. total end-to-end latency

Trace the current Gradio handlers/core functions rather than reimplementing algorithms.

## Required experiments

Use a deterministic representative example based on the existing demo defaults.

Prefer initially:

```text
prompt: a dolphin with wings
segment prompt: dolphin
seed: 42
steps: 4
```

Adjust only if the current released interface requires a slightly different valid configuration, and document the adjustment.

### Experiment A — 80GB mode

Fresh process:

* one cold run
* three warm runs
* fixed prompt
* fixed seed
* 4-step mode

### Experiment B — forced 48GB mode

Fresh process:

* same prompt
* same seed
* same model settings
* one cold run
* three warm runs

Do not compare a warm process against a cold one.

### Experiment C — model residency

Measure GPU memory after loading models in the natural pipeline order.

Determine whether the required Qwen/SAM3/MoGe/tiling bundles can coexist on one A100 80GB.

If not, measure explicit unload strategies using the existing model-manager unload methods.

Do not implement a shared-Qwen architectural refactor until these measurements exist.

### Experiment D — 4-step vs 40-step

Only after the basic 4-step path works.

Measure the latency difference and note the likely public-demo default.

## External GPU trace

While the benchmark is active, capture approximately once per second:

* GPU utilization
* memory utilization
* used/free VRAM
* power draw

Use `nvidia-smi` and save the trace under `profiling/results/`.

## Required deliverables

Write raw machine-readable results and:

```text
profiling/results/PROFILE_SUMMARY.md
```

The summary must include:

* environment and Git SHA
* cold/warm stage timings
* 80GB vs 48GB comparison
* peak/resident VRAM
* model-load times
* GPU-utilization observations
* any OOM or model-residency constraints

Explicitly answer:

```text
Primary bottleneck:
Secondary bottleneck:
Warm end-to-end latency:
80GB vs 48GB speedup:
Peak VRAM:
Can all necessary models coexist:
Time attributable to model loading/offloading:
Recommended public-demo GPU configuration:
Recommended public-demo inference configuration:
Recommended next optimization:
```

## Completion criteria

Do not declare the task complete merely because the profiling script exists.

The task is complete when:

1. the released environment has been reproduced,
2. at least one representative full inference path has executed successfully, or a concrete unavoidable blocker is documented,
3. the required 80GB/48GB experiments have actually been run,
4. results have been saved,
5. `PROFILE_SUMMARY.md` contains evidence-based conclusions.

Do not begin implementing the simplified public demo in this task. Profiling comes first.
