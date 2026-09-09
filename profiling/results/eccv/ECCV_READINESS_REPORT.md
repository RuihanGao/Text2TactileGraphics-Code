# ECCV readiness report — 2026-09-08

> Historical verification report. At release handoff on 2026-09-09, the operator
> confirmed external laptop and phone access on the hardened dual-GPU service.
> The original local measurements and then-pending external checks below are
> preserved as recorded; no new external benchmark or QR-redirect test is claimed.
> Current deployment instructions: [ECCV_DEMO.md](../../../ECCV_DEMO.md).

The dual-A100 resident lifecycle meets the local warm latency target with unchanged true-80GB reference outputs. Public access is **not validated**: the same-Pod proxy recheck returned HTTP 403 for both the page and readiness endpoint. RunPod HTTP exposure and an actual outside-client generation remain operator actions.

## Deployment

- Pod/GPU configuration: prepared RunPod, exactly **2× NVIDIA A100-SXM4-80GB**; driver 550.127.05, PyTorch 2.11.0+cu128, CUDA runtime 12.8, Python 3.12.14. Cgroup RAM limit 465.66 GiB; CPU quota 54.4 cores. Frozen offline environment reproduced.
- Base Git SHA: `f2975d096d63c02e9edf0fa19493028ae4ff33c6`, branch `deploy/eccv-demo-2gpu`; local uncommitted hardening changes, no push. Runtime source checksums saved with each measured process. This prepared deployment branch supersedes the older profiling-branch instruction as documented in the setup report.
- Lifecycle: `dual_a100_80gb`, true `80gb` numerical mode, released checkpoints and unchanged resolution/steps (normal route 4/4/10). No quantization, architecture, checkpoint or topology changes.
- GPU0: resident base-edit Qwen, SAM3 and MoGe. GPU1: one resident Qwen-Image base shared by texture and tiling, with serialized validated adapter switching.
- Startup warmup: **308.11 s** for the final process, including actual backend inference before Ready. Other clean cache-dependent launches took 103–314 s; do not use a fixed sleep as readiness.
- Public port: application binds **0.0.0.0:8080**, one worker. Proxy address derives from `RUNPOD_POD_ID`; no source-code Pod ID.
- External validation status: **WAITING_FOR_EXTERNAL_VALIDATION**. Same-Pod proxy checks returned 403/404, not success. Local queued SSE progress and final mesh download passed. See `MANUAL_ACTION_REQUIRED.md` in the checkout root.
- Final service remains running from `launch_eccv.sh`; recorded PID file `/tmp/text2tactilegraphics-eccv.pid`. Current log: `profiling/results/eccv/stage4-readiness/cleanup-server.log`; queue log: `cleanup-events.jsonl`; timing observer: `cleanup-service/stages.jsonl`. Fault injection is disabled. `/readyz` is lightweight and does not load models.

## Latency

Final configuration: ten sequential real mobile-browser requests in one warmed process, live structured Gemini planner and real mesh export. Full latency measures Generate click through UI Done; download validity is checked separately. Backend is the sum of disjoint actual pipeline handlers, including all generation, segmentation and geometry work; planner and UI/transport overhead are separate. One extension request overlapped the final CPU test suite, so reported timings retain that incidental load.

| Measurement | p50 | Mean | Min–max |
|---|---:|---:|---:|
| Full Quick Trial | **45.056 s** | **45.666 s** | 44.186–48.151 s |
| Backend | **38.479 s** | 38.808 s | 37.686–40.718 s |
| Structured planner | **2.050 s** | 2.268 s | 1.829–3.969 s |
| Adapter switching | 0.966 s | 0.978 s | 0.940–1.053 s |
| CPU geometry spans | 25.047 s | 25.326 s | 24.350–27.211 s |
| Request-end cycle collection | 0.365 s | 0.370 s | 0.351–0.421 s |

- Worst observed final warm run: **48.151 s**. Target mean <60 s: **YES locally**. No p95 estimate from ten observations; external network performance remains unmeasured.
- Remaining large Qwen load per accepted warm request: **zero**. Small texture adapters are reapplied; adapter time is included in backend timing, not an additional total.
- Primary remaining bottleneck: CPU mesh construction/displacement/export. Next optimization should profile and improve that implementation without changing geometry outputs; no unrelated topology repair was attempted.
- Planner retained: six live schema/semantic cases passed at about 2.3 s median, including dolphin, multi-region lamp, one-region examples and textureless input. Faster-provider substitution was not justified.

## Capacity

- Active requests: **1**, protected by an entire-workflow lock in addition to the Gradio queue. Both GPUs serve one request; adapter state must not overlap across requests.
- Maximum pending: **2**, with a projected wait budget of **180 s** and expiring pending tickets. Admission occurs before planner/GPU work. Environment settings are in the runbook/launcher.
- Expected representative service time: 45–50 s. One job ahead: roughly one minute; two ahead: roughly 1–2 minutes.
- Measured burst throughput: **about 75 successful requests/hour**. Long multi-region or 40-step advanced requests can take longer; rolling service duration reduces admission when projected waits exceed budget.
- Real 1/2/4/6-client escalation: **9 accepted, 9 completed, 4 friendly overload rejections, zero accepted failures**, maximum server queue wait **92.16 s**, maximum active workflows **1**.
- Overload behavior: persistent visible retry message, no planner/GPU work. Browser/session duplicate prevention survives refresh and cross-tab Gradio session changes. Direct unqueued expensive API execution is disabled; queued work requires an owner-bound admission ticket.
- Telemetry records queue/wait/outcome/error class without prompt contents. Local traffic tests passed; public-proxy traffic tests remain pending.

## Stability and numerical behavior

- Final soak: **10/10 successful real requests**, zero browser errors, all ten GLBs byte-identical. Prior final-process five-request iteration and nine accepted traffic requests also completed successfully.
- Reference equivalence: dolphin plus two lamp replicates match saved single-A100 true-80GB base/mask/texture/normal/tile artifacts exactly. Final GLBs also match those fixed-plan references byte-for-byte. Live planner cohorts use their actual structured plan and are checked for repeatability separately from fixed-plan equivalence.
- Peak allocated GPU0/GPU1: **61.206 / 57.149 GiB**; peak reserved **62.658 / 58.707 GiB**. GPU residency settled after request two at **59.134 / 53.816 GiB**, exactly constant through request ten. No progressive VRAM growth or CUDA OOM observed.
- 1 Hz trace: mean GPU utilization **12.36% / 11.38%**, both peak 100%; sampled NVML peak used **62.153 / 57.233 GiB**. Sampling can miss brief allocator peaks. Low average utilization reflects serial GPU stages and substantial CPU geometry, not repeated Qwen loading.
- Host memory: the initial five-request iteration rose from 4.87 to 5.82 GiB at handler completion. Request-end `gc.collect()` now reclaims geometry reference cycles without evicting models. Final post-collection RSS was 4.19 GiB after request one, 4.69 after five, and fluctuated **4.81–5.19 GiB over the last five**, ending **5.05 GiB**. The earlier continuous rise did not persist; this short soak cannot prove absence of a long-duration host-memory leak. Continue watching RSS during the poster session.
- Failure recovery: controlled planner error, generation exception, invalid empty prompt, full queue and disconnected browser all released admission and allowed later work. Actual browser closure during GPU generation recorded cancellation, released its slot in about five seconds, and a following real generation/download succeeded. Injected failures are separate from the zero unexpected failures in accepted traffic/soak requests.
- Restart validation: clean launch observed Warming, HTTP 503 and disabled Generate; Ready followed complete model/kernel/geometry warmup. The first accepted request was warm. Launcher rejects insufficient GPUs and occupied ports; safe-stop helper rejects unrelated processes. Single-A100 fallback configuration and existing lifecycle tests pass; earlier validated single-A100 artifacts remain unchanged.
- Mobile: 390×844 browser, no horizontal overflow, visible service/busy status and Generate, usable rendered 3D viewer/download, customization collapsed by default. Physical phone/cellular validation is pending.
- Real customization regression: changing the Braille label rebuilt only geometry/final mesh in **27.629 s**, with no planner or image regeneration. The first harness attempt acted before the final form update settled; its apparent pass was rejected and the corrected test verified the actual edited label and stage trace (`edit-check-v2/result.json`).
- Final core regression: **274 passed, 27 deselected**; changed-code Ruff checks, shell syntax and Git whitespace checks passed. Credential-value scan found no matches in changed files/reports/logs; attempted private credential-file download was denied with HTTP 403, without reading its body.

## Evidence

Stage reports: `stage0-environment/REPORT.md`, `stage1-dual-gpu/REPORT.md`, `stage2-public-access/REPORT.md`, `stage3-traffic/REPORT.md`, `stage4-readiness/REPORT.md`. Raw measurements remain on the persistent workspace and are gitignored. Final metrics: `stage4-readiness/final-metrics.json`, `final-utilization.json`, `cleanup-soak/`, `cleanup-soak-extension/`, `cleanup-service/`, `cleanup-events.jsonl`, `cleanup-gpu.csv`; failure evidence: `failure-check.json`, `recovery-verification.json`. Stage 1 includes reference comparison manifests, planner tests and final-mesh equivalence. Stage 3 includes CSV/JSON client results, admission telemetry and GPU traces for each escalation.

## Friday operator runbook

1. **Start the Pod.** In RunPod, start the prepared Pod with its persistent model-cache volume and exactly 2× A100-SXM4-80GB. Expose internal **HTTP 8080**. SSH into the Pod; keep the private Gemini credential file available. No credential should be pasted into logs or source.

2. **Launch the service.**

   ```bash
   cd /workspace/Text2TactileGraphics-Code-2gpu
   ./launch_eccv.sh --check
   mkdir -p profiling/results/eccv/live
   nohup env ECCV_MEASUREMENT_DIR=profiling/results/eccv/live TEXT2TACTILEGRAPHICS_TELEMETRY=profiling/results/eccv/live/events.jsonl ./launch_eccv.sh > profiling/results/eccv/live/server.log 2>&1 &
   ```

   The launcher records its PID in `/tmp/text2tactilegraphics-eccv.pid`, uses one application worker, fixes `0.0.0.0:8080`, validates GPUs/caches, and enables the measured poster settings. It refuses an occupied port and never stops another process. Instrumentation is optional: omit `ECCV_MEASUREMENT_DIR` to run the application entry point directly. Do not set `ECCV_TEST_FAULTS` during normal operation.

3. **Wait for Ready.**

   ```bash
   curl -s -w '\nHTTP %{http_code}\n' http://127.0.0.1:8080/readyz
   tail -n 40 profiling/results/eccv/live/server.log
   ```

   Repeat until HTTP 200 and `ready: true`. HTTP 503 with Warming is expected during model loading and full startup inference. An Unhealthy result requires reading the log and resolving the reported error, then restarting. Observed cache loads vary by minutes; wait for Ready rather than a fixed timer.

4. **Test from the phone.** Obtain the current destination:

   ```bash
   echo "https://${RUNPOD_POD_ID}-8080.proxy.runpod.net"
   ```

   Disable phone Wi-Fi and use cellular. Open the URL, check Quick Trial and Ready, generate a dolphin, wait for Done, manipulate the 3D viewer and download the mesh. Also run the laptop command in `MANUAL_ACTION_REQUIRED.md` and retain its JSON/screenshot. A localhost or same-Pod proxy test does not replace this check.

5. **Check GPU state.**

   ```bash
   nvidia-smi -L
   nvidia-smi --query-gpu=index,memory.used,memory.free,utilization.gpu,power.draw --format=csv
   ```

   Both large Qwen bases should remain resident after Ready. CUDA allocated residency is about 59.1/53.8 GiB; NVML additionally includes contexts/reserved allocations. Large Qwen load messages should occur only during startup.

6. **Check the queue.**

   ```bash
   curl -fsS http://127.0.0.1:8080/healthz
   ```

   Expect at most one active workflow and two pending jobs. A `scheduled` slot denotes the accepted next-to-run job before execution begins. The UI shows a coarse wait estimate and persistent overload feedback. Do not increase GPU concurrency; both GPUs serve one workflow.

7. **Read logs.**

   ```bash
   tail -n 80 profiling/results/eccv/live/server.log
   tail -n 30 profiling/results/eccv/live/events.jsonl
   tail -n 10 profiling/results/eccv/live/stages.jsonl
   ```

   Queue telemetry excludes prompt contents. Expected overload is `rejected`; investigate unexpected `failure`, `OutOfMemoryError`, or a long-lived active slot. The current measured service's corresponding files are listed in the deployment section above.

8. **Restart only the demo.** Prefer waiting for the queue to drain, then:

   ```bash
   .venv/bin/python scripts/stop_eccv.py
   ```

   This verifies the recorded process command and working directory before sending SIGTERM. It refuses other processes and does not force-kill a draining service. Repeat the launch in step 2, wait for Ready and repeat the phone check. Do not use a blanket Python/process kill command.

9. **Switch to the single-A100 fallback.** Stop the identified demo with step 8, then:

   ```bash
   ./launch_eccv.sh --single --check
   nohup env ECCV_MEASUREMENT_DIR=profiling/results/eccv/live-single TEXT2TACTILEGRAPHICS_TELEMETRY=profiling/results/eccv/live-single/events.jsonl ./launch_eccv.sh --single > profiling/results/eccv/live/fallback.log 2>&1 &
   ```

   This selects GPU 0 and the existing `single_a100_80gb` unload lifecycle with true 80GB numerics. Set `TEXT2TACTILEGRAPHICS_SINGLE_GPU=1` before launching if GPU 1 is the desired fallback. It is slower and reloads Qwen between stages/requests; the dual-residency latency result does not apply. The original advanced interface and fallback checkout remain preserved.

10. **Update the stable QR redirect.** After external validation succeeds, sign in to the existing redirect provider and change the stable printed QR destination to the current Pod's 8080 proxy URL. Scan the printed QR on cellular and complete a generation again. The redirect provider is an operator action outside this Pod; no redirect change has been claimed.

## Final decision

READY FOR POSTER: NO

Unresolved blocker: expose/verify RunPod HTTP 8080, then complete and record a real outside-laptop and phone/cellular generation. Update and test the stable QR redirect after that succeeds. Exact actions are in `MANUAL_ACTION_REQUIRED.md`.
