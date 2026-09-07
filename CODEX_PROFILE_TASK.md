Read `AGENTS.md` completely before doing any work.

Carry the RunPod A100 profiling task through to completion. Work autonomously on the current `profile/runpod-a100` branch.

Start by inspecting the current repository and reproducing the released environment exactly. Verify the machine, Git state, Python/PyTorch/CUDA environment, persistent cache configuration, custom checkpoints, and relevant tests. Use Qwen rather than Gemini for the baseline so that no Gemini API key is required.

Then trace the current Gradio handlers and core generation classes to identify the real released end-to-end execution path. Do not create a fake benchmark that merely instantiates individual models. Build `profiling/profile_pipeline.py` around the actual generation functions used by the application.

Run the required measurements described in `AGENTS.md`:

* model-load timing and resident VRAM
* per-stage synchronized CUDA timing
* one cold and three warm 80GB-mode runs
* one cold and three warm forced-48GB-mode runs in a fresh process
* model residency/lifecycle measurement
* 4-step vs 40-step comparison only after the 4-step path works
* background `nvidia-smi` utilization/memory trace

Save raw results under `profiling/results/` and create `profiling/results/PROFILE_SUMMARY.md`.

Bias toward action and follow through. Resolve ordinary dependency, path, cache, and implementation issues yourself. Do not stop merely to describe what should be done. Do not ask for confirmation for reversible repository edits, tests, downloads, profiling runs, or cleanup needed to complete this task.

Do not push Git commits, modify the upstream repository, change model quality, rewrite the serving architecture, or build the public demo yet.

If something genuinely blocks the full benchmark, first complete all profiling that is still possible, preserve the logs and partial measurements, and document the precise blocker and the next command/action required.
