# Stage 0 — PASS

Hypothesis: the isolated deployment environment reproduces the released lockfile and has exactly two suitable GPUs. Confirmed: two NVIDIA A100-SXM4-80GB, driver 550.127.05, Python 3.12.14, Torch 2.11.0+cu128 / CUDA 12.8. CUDA allocation and synchronization pass on both devices. See `environment.json` for immutable SHA, branch, RAM/cgroup, CPU quota, disk and cache paths.

`uv sync --frozen --offline` passes; existing core/focused suites pass: 264 passed, 27 slow deselected. No application changes preceded these checks. Initial tracked application diff was empty. Required reports and application lifecycle/Quick Trial sources were inspected. Custom and released cache paths were verified in setup; reference artifacts are available read-only under `/workspace/Text2TactileGraphics-Code/profiling/results/single-gpu-quality/`. Gemini private credential file and inherited HF credential are present; no values recorded.

Reuse those saved validated references for Stage 1; full inference on the new driver remains to be established there. Initial environment recorder assumed cgroup v2; corrected to the actual cgroup v1 paths, with no environment changes. Work stays in the isolated prepared deployment branch, as established by SETUP_VERIFICATION_2GPU.md, superseding the earlier profiling-only scope.
