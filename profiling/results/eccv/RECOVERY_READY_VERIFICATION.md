# ECCV Hardened Service Recovery Verification

> Historical verification report. At release handoff on 2026-09-09, the operator
> confirmed external laptop and phone access on the hardened dual-GPU service.
> The original local measurements and then-pending external checks below are
> preserved as recorded; no new external benchmark or QR-redirect test is claimed.
> Current deployment instructions: [ECCV_DEMO.md](../../../ECCV_DEMO.md).

## Overall
HARDENED SERVICE READY LOCALLY: YES

Verified on 2026-09-09 UTC. Existing hardened implementation used without application or dependency changes.

## Repository
Path: /workspace/Text2TactileGraphics-Code-2gpu
Branch: deploy/eccv-demo-2gpu
Git SHA: f2975d096d63c02e9edf0fa19493028ae4ff33c6
Working-tree status: Existing staged hardening changes and untracked CODEX_VERIFY_ECCV_RECOVERY_READY.md preserved. Full initial status recorded in recovery-environment.json. No reset, checkout, cleanup, stash, commit or push performed.
Hardened files present: YES — launch_eccv.sh and scripts/stop_eccv.py.
dual_a100_80gb present: YES — configuration, model manager, inference service and launcher.
health/readiness endpoints present: YES — src/text2tactilegraphics/ui/public_app.py.
Prior reports retained: ECCV_READINESS_REPORT.md and stage4-readiness/REPORT.md.

## Environment
Python: 3.12.14 (existing .venv works).
PyTorch: 2.11.0+cu128.
CUDA: PyTorch runtime 12.8; available YES; NVIDIA driver 550.127.05 (nvidia-smi displays CUDA 12.4).
GPU count: 2.
GPU 0: NVIDIA A100-SXM4-80GB; 79.14 GiB available to PyTorch.
GPU 1: NVIDIA A100-SXM4-80GB; 79.14 GiB available to PyTorch.
Environment repair required: NO.

## Caches / credentials
HF_HOME: /workspace/model_cache/huggingface
TEXT2TACTILEGRAPHICS_CKPT_DIR: /workspace/model_cache/text2tactilegraphics/ckpt
DIFFSYNTH_MODEL_BASE_PATH: /workspace/model_cache/diffsynth
HF_TOKEN present: YES.
Gemini env present: YES — nonempty /workspace/private/gemini.env; launcher validated and exported the planner credential.
No secret values printed: YES.
The three cache variables were unset in the incoming shell. The existing launcher exported the expected defaults; all directories exist and their effective paths were verified in the running service. No configuration edits required.

## RunPod / port
RUNPOD_POD_ID: <POD_ID>
RUNPOD_VOLUME_ID: <VOLUME_ID> (matches expected persistent volume).
Port 8080 before launch: available; checked with ss and a successful bind probe. Also available before retry.
Expected public URL: https://<POD_ID>-8080.proxy.runpod.net
External access verified: NO.
RunPod control-plane HTTP exposure was not independently verified; the URL is derived only.

## Launcher preflight
Status: PASS (exit 0), exactly two A100 GPUs, dual_a100_80gb, true 80gb mode, cache/checkpoint and credential configuration validated.
Log: profiling/results/eccv/recovery-preflight.log.
Issues fixed: No environment/dependency fixes needed. Cache defaults supplied by launcher.

## Launch
Launch command:
```bash
nohup env ECCV_MEASUREMENT_DIR=profiling/results/eccv/live TEXT2TACTILEGRAPHICS_TELEMETRY=profiling/results/eccv/live/events.jsonl ./launch_eccv.sh >> profiling/results/eccv/live/server.log 2>&1 < /dev/null &
service_pid=$!
echo "$service_pid" > profiling/results/eccv/live/nohup-shell.pid
wait "$service_pid"
```
Service PID: 10632 (matches /tmp/text2tactilegraphics-eccv.pid and the 8080 listener).
Server log: profiling/results/eccv/live/server.log.
Lifecycle: dual_a100_80gb.
Bind address/port: 0.0.0.0:8080.
The first background attempt (PID 10307) exited before emitting startup output. No service process or listener remained; cause was not established. Relaunched with a retained command session using shell wait; the service then started successfully. No process was killed and no concurrent instance was launched. The live-output directory was absent before this task.
Service left running.

## Readiness
First observed state: connection unavailable before bind, then HTTP 503 / Warming / ready=false at 04:08:12 UTC.
Time to Ready: application warmup 309.98 s; approximately 343.7 s from the successful launch to the first Ready poll; 407.1 s from the initial attempt. Poll interval 5 s, bounded to 15 minutes. One transient request timeout occurred during loading and subsequent polls succeeded.
Final /readyz HTTP status: 200 (first observed 04:13:19 UTC).
Final ready value: true; state Ready; lifecycle dual_a100_80gb; gpu_count 2.
Final /healthz status: HTTP 200; ready=true; empty queue.
Warmup completed base generation, segmentation, texture/tiling and final mesh with one texture and one Braille annotation according to server log and stage telemetry. No additional generation or load tests were issued.
Evidence: live/recovery-readiness.jsonl, live/recovery-final.json, live/recovery-launch.json, live/stages.jsonl.

## GPU residency after Ready
GPU0 used memory: 62853 MiB = 61.380 GiB NVML; free 18185 MiB.
GPU1 used memory: 55909 MiB = 54.599 GiB NVML; free 25129 MiB.
Final recorded warmup-stage CUDA allocated memory: GPU0 59.116 GiB; GPU1 53.807 GiB. NVML includes allocator/context overhead. Both resident model stacks remain populated, consistent with prior hardened residency.
OOM observed: NO — no CUDA OOM in startup log or stage errors; warmup and readiness succeeded.

## Blockers
- None for local readiness.
- External RunPod HTTP exposure/reachability and laptop/phone validation remain unverified and outside this task.

## Final conclusion
HARDENED SERVICE READY LOCALLY: YES

The next operator action is external RunPod HTTP validation from the laptop and phone/cellular network.
