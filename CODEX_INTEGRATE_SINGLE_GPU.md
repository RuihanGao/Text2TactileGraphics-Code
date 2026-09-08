Read `AGENTS.md`, `profiling/results/PROFILE_SUMMARY.md`, and `profiling/results/SINGLE_GPU_DEPLOYMENT_REPORT.md` before modifying application code.

## Mission

Integrate the validated single-A100 inference lifecycle into the actual Text2TactileGraphics Gradio application.

The profiling/validation experiments established that:

* one NVIDIA A100-SXM4-80GB is sufficient;
* all Qwen generation stages must retain the repository's true `80gb` numerical behavior;
* forced `48gb` mode is NOT the desired deployment path;
* natural 80gb model caching OOMs;
* true `80gb` mode with targeted sequential Qwen unloading reproduces the required outputs;
* validated cold latency is about 97 s;
* validated warm latency is about 75 s;
* peak VRAM is about 64 GiB;
* remaining Qwen loading overhead is about 29 s.

Do not repeat the capacity investigation unless needed to validate the application integration.

## Goal

Make the actual released Gradio workflow runnable on one A100 80GB using:

`true 80gb inference + targeted Qwen lifecycle`

without requiring the profiling harness.

The existing advanced application:

`src/text2tactilegraphics/ui/app.py`

must remain functionally intact.

## Required lifecycle

Use the same lifecycle already validated by the profiling harness.

Conceptually:

1. Load/use `qwen_base_edit` in true 80gb mode.
2. Generate the base image.
3. Release all references that prevent `qwen_base_edit` from being reclaimed.
4. Use the existing `ModelManager.unload_model(...)`.
5. Preserve SAM3/MoGe where appropriate.
6. Load/use `qwen_texture` in true 80gb mode.
7. Generate tactile texture.
8. Fully unload `qwen_texture`.
9. Load/use `qwen_tiling` in true 80gb mode.
10. Generate tileable texture.
11. Fully unload `qwen_tiling`.

Do not alter model dtype, checkpoints, LoRAs, schedulers, steps, CFG settings, or numerical behavior.

Do not silently fall back to `48gb`.

## Implementation approach

First inspect how the profiling harness implemented the validated `unload-qwen` lifecycle.

Reuse the smallest possible mechanism in the application/core model lifecycle rather than duplicating profiling-specific logic.

Prefer a clearly named configuration such as:

`single_a100_80gb`

or an equivalent explicit lifecycle option.

If the cleanest implementation is to modify `ModelManager` or its callers, do so minimally.

The existing research UI should still support its existing configurations where feasible.

## Safety checks

After each Qwen unload:

* verify Python references owned by LoRA managers/pipelines do not keep the model alive;
* call the existing unload mechanism;
* record allocated/reserved VRAM in debug logs;
* do not depend on forced CPU offload.

Avoid unnecessary `torch.cuda.empty_cache()` calls unless profiling evidence shows they are required for successful lifecycle transitions.

## Regression validation

After integration:

### 1. Tests

Run the existing core tests.

### 2. Numerical validation

Run the fixed-seed single-GPU equivalence check again through the actual application/core path, not only the profiling harness.

Confirm that application outputs match the previously validated targeted-unload 80gb outputs.

### 3. Representative examples

Run:

`a dolphin with wings with an avocado skin texture.`

and:

`lamp, the base of the lamp has a tree bark texture, and the lamp shade has a cloth_bag texture.`

using the actual application handlers.

Save output artifacts for manual inspection.

Do not spend time improving the documented lamp segmentation limitation in this task.

### 4. Memory

Verify the complete application workflow stays below the A100 80GB limit.

### 5. Latency

Measure at least one cold and two warm application runs.

Compare against the validated profiling target of approximately 75 s warm.

A modest application/UI overhead is acceptable.

## Gradio launch

Once integration passes validation, launch the existing application on:

* host: `0.0.0.0`
* port: `7860`

Use the current project's normal Gradio entry point.

Do not create the simplified public UI yet.

## Required report

Create:

`profiling/results/APPLICATION_INTEGRATION_REPORT.md`

Answer:

* Does the actual Gradio/core application use true 80gb numerical mode?
* Does it complete on one A100?
* Peak VRAM?
* Cold application latency?
* Warm application latency?
* Are representative outputs consistent with validation?
* What files were modified?
* What remains before a public demo?

Do not push commits.
Do not add multi-GPU support.
Do not quantize or reduce quality.
Do not redesign the application.



