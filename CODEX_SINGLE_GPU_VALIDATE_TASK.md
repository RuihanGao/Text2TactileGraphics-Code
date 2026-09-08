Read `AGENTS.md` and `profiling/results/PROFILE_SUMMARY.md` completely before doing any work.

## Mission

Complete the profiling and output-quality validation required to decide whether **one NVIDIA A100-SXM4-80GB** is sufficient for deployment of the current Text2TactileGraphics repository.

Do **not** use multiple GPUs in this task.

Do **not** use forced `48gb` mode as the deployment-quality reference. Previous project experience indicates that the 48GB/offload path can produce visibly worse results because of its numerical/model-weight behavior.

Our target configuration is:

**true `80gb` numerical/model behavior + targeted sequential Qwen unloading on one A100 80GB.**

The purpose of targeted unloading is only to manage model residency. It must not change the model weights, dtype, inference configuration, LoRAs, scheduler, seed behavior, or outputs relative to a true isolated 80GB execution of each stage.

The existing profiling baseline showed:

* natural `80gb` caching OOMs while loading later Qwen models;
* targeted-unload `80gb` completes on one A100 80GB;
* targeted-unload `80gb` warm E2E is approximately 81 seconds;
* forced `48gb` is substantially slower;
* repeated Qwen model loading is the dominant remaining single-GPU latency cost.

Treat those measurements as the existing baseline rather than repeating work unnecessarily.

## Primary questions to answer

At the end of this task, explicitly answer:

1. Can the complete released pipeline run reliably on **one A100 80GB** while every Qwen inference stage uses the true `80gb` execution path?
2. Does targeted unloading preserve the output of isolated true-80GB execution for:

   * base-image generation,
   * texture generation,
   * tiling?
3. Does the targeted-unload pipeline reproduce visually good results on representative examples from the project?
4. What is the best reproducible cold and warm single-GPU latency?
5. What is the peak VRAM of the final recommended single-GPU lifecycle?
6. What model-loading time remains after optimization?
7. Based on these results, is there any **quality or capacity reason** to rent multiple GPUs, or would multiple GPUs only be a latency optimization?

Do not recommend multi-GPU deployment until these questions are answered experimentally.

---

## A. Validate true-80GB numerical equivalence

Do not use forced `48gb` mode for these reference comparisons.

For each Qwen stage:

1. `qwen_base_edit`
2. `qwen_texture`
3. `qwen_tiling`

construct an **isolated 80gb reference execution** in a fresh Python process with only the models necessary for that stage resident.

Then compare it against the same stage when executed as part of the targeted-unload `80gb` lifecycle.

Use exactly the same:

* checkpoint and revision,
* custom checkpoint files,
* model dtype,
* model device placement,
* LoRA/adapters and weights,
* scheduler,
* guidance/CFG settings,
* inference steps,
* image size,
* prompt,
* input artifact,
* seed.

Record the actual runtime configuration rather than relying only on config-file assumptions.

For every major model, report at least:

* relevant parameter dtype,
* device,
* checkpoint identity,
* LoRA identity and scale,
* scheduler,
* inference steps,
* CFG/guidance,
* seed.

### RNG validation

Verify that each stochastic generation stage uses an explicitly controlled seed/generator so that unloading and reloading models does not accidentally shift a global RNG stream.

If necessary, instrument the existing generation code for validation, but do not alter the released generation semantics.

### Output comparison

For every isolated-80GB vs targeted-unload-80GB pair:

1. save both outputs;
2. compute SHA256 of the saved image;
3. if hashes differ, compute at least:

   * maximum absolute pixel difference,
   * mean absolute pixel difference;
4. explain the first operation at which divergence appears.

Exact equality is preferred.

Do not conclude that `48gb` and `80gb` are equivalent merely because one particular previous benchmark happened to produce identical PNG hashes.

The comparison of interest is:

**isolated true 80gb vs targeted-unload true 80gb.**

---

## B. Quality-check representative project examples

Run the actual released generation workflow using the final recommended single-A100 `80gb + targeted unload` lifecycle.

Use project-representative examples rather than only synthetic profiling prompts.

### Example 1 — dolphin

Use the semantic request:

`a dolphin with wings with an avocado skin texture.`

Reproduce the intended released pipeline interpretation using the repo's actual handlers and input structure.

The intended semantic content is:

* object/scene: dolphin with wings
* tactile texture: avocado skin

Use the existing released mechanism for specifying the base-image prompt, segmentation target, and tactile-texture prompt rather than inventing a new interface.

Use a fixed deterministic seed and record all settings.

Save at least:

* generated base image,
* segmentation result,
* generated tactile texture/normal output,
* tileable texture,
* final generated tactile graphic/mesh.

### Example 2 — multi-texture lamp

Use the semantic request:

`lamp, the base of the lamp has a tree bark texture, and the lamp shade has a cloth_bag texture.`

The intended content is:

* object: lamp
* region 1: lamp base
* texture 1: tree bark
* region 2: lamp shade
* texture 2: `cloth_bag`

Use the **existing released multi-region / repeated segmentation-and-texture workflow** if the repository supports it.

Do not simplify this example to one texture merely to make the benchmark easier.

Determine from the existing UI/handlers how the project currently assigns textures to multiple semantic regions, and reproduce that workflow faithfully.

Save intermediate outputs for both regions and the final combined tactile result.

If the released code cannot directly express this example, document the exact limitation rather than silently changing the task.

### Quality-check goal

This is not intended to become a formal perceptual evaluation.

The goal is to establish that the single-GPU targeted-unload lifecycle can reproduce the kinds of outputs that motivated the original 80GB-quality configuration.

Preserve outputs under a clear directory such as:

`profiling/results/single-gpu-quality/`

Include a small HTML or Markdown contact sheet/index if useful for manual visual inspection.

Do not alter checkpoints, quantize models, reduce resolution, reduce quality settings, or substitute models to improve speed.

---

## C. Complete single-GPU performance profiling

Once numerical equivalence has been established, benchmark the final recommended **one-A100, true-80GB, targeted-unload** lifecycle.

Use fresh processes where required.

Measure:

### Cold

* at least one fresh-process cold run.

### Warm

* at least three warm runs;
* same representative settings;
* report mean, min, max.

Record:

* end-to-end latency,
* model-load time,
* per-Qwen model load time,
* inference time,
* explicit unload time,
* CPU geometry time,
* allocated VRAM peak,
* reserved VRAM peak,
* sampled NVML peak,
* process/host memory where useful.

Use synchronized CUDA timing for GPU stages.

Retain approximately 1 Hz `nvidia-smi` traces.

The existing profiling harness should be reused and extended rather than rewritten unnecessarily.

---

## D. Investigate the safest obvious single-GPU lifecycle improvement

The previous profiling result showed approximately 37.7 seconds of Qwen model initialization/loading in the fastest warm lifecycle.

Before considering multiple GPUs, investigate whether `qwen_texture` and `qwen_tiling` use sufficiently compatible Qwen-Image base weights that the same base pipeline could potentially remain resident while switching only the correct LoRA/configuration/state.

First inspect and report:

* base checkpoint identity,
* pipeline classes,
* transformer/text components,
* LoRAs/adapters,
* scheduler state,
* any mutable pipeline state,
* memory residency.

Do **not** implement model sharing unless compatibility is clear.

If a minimal and safe sharing/reuse strategy can be implemented without changing numerical outputs, test it separately.

Any optimization must pass the same isolated-80GB output-equivalence checks.

If exact equivalence cannot be guaranteed, keep the already-proven targeted-unload lifecycle as the deployment recommendation.

Do not attempt quantization or lower precision.

---

## E. Geometry/export handling

Do not spend this task debugging general mesh watertightness.

The project already has downstream/off-the-shelf mesh-cleaning procedures that can be used where necessary.

Record export validity as diagnostic information, but a non-watertight raw mesh is **not a blocker for this deployment profiling task**.

Focus on:

* model output quality,
* true-80GB numerical behavior,
* VRAM capacity,
* latency,
* lifecycle.

---

## F. Required final report

Create:

`profiling/results/SINGLE_GPU_DEPLOYMENT_REPORT.md`

Include:

### Numerical equivalence

For each:

* base image,
* texture generation,
* tiling,

report:

* isolated 80GB output hash,
* targeted-unload 80GB output hash,
* exact match yes/no,
* numeric difference if applicable,
* config/dtype/device equality.

### Representative quality examples

Include the saved artifacts/paths for:

1. `a dolphin with wings with an avocado skin texture.`
2. `lamp, the base of the lamp has a tree bark texture, and the lamp shade has a cloth_bag texture.`

State whether each workflow completed correctly.

### Performance

Report:

* cold E2E latency,
* three warm latencies,
* warm mean,
* peak allocated VRAM,
* peak reserved VRAM,
* Qwen loading time,
* actual generation/inference time,
* CPU geometry time.

Compare against the previous ~81 s targeted-unload baseline.

### Deployment conclusion

Answer exactly:

`Can one A100 80GB reproduce true-80GB-quality outputs?`

`Does targeted unloading change numerical outputs?`

`Recommended single-GPU lifecycle:`

`Warm single-GPU latency:`

`Peak VRAM:`

`Remaining Qwen loading overhead:`

`Can the current repository be deployed on one A100 80GB without using 48gb numerical mode?`

`Is more GPU memory required for quality/correctness?`

`Would renting additional GPUs provide capacity, quality, latency, or only concurrency benefits?`

`Recommended next experiment before renting multiple GPUs:`

---

## Constraints

* Use only the single A100 80GB available on this RunPod.
* Do not change model checkpoints.
* Do not use quantization.
* Do not lower precision to save memory.
* Do not switch to forced `48gb` mode for the deployment candidate.
* Do not reduce image resolution or inference quality settings.
* Do not modify the upstream remote.
* Do not push anything.
* Do not build the public Gradio app yet.
* Do not spend substantial time repairing mesh topology.
* Preserve raw evidence and output images.
* Resolve routine reversible implementation issues autonomously.
* If a genuine blocker occurs, complete all remaining useful experiments and document the exact blocker.

The task is complete only when we have enough evidence to decide whether to stay with one A100 80GB or rent additional GPUs specifically for speed.
