# Stage 4 — PASS (local readiness; external validation pending)

Hypothesis: explicit startup warmup, owner-bound admission and whole-workflow serialization preserve stable resident inference across restart, traffic and recoverable failures.

- Full core regression: **273 passed, 27 deselected** before the final memory-cleanup refinement.
- Deterministic `launch_eccv.sh`: dual and single fallback configuration checks pass; insufficient dual-GPU hardware and occupied port fail clearly. It fixes one application worker, explicit lifecycle, true-80GB hardware, poster limits and `0.0.0.0:8080`.
- Verified stop helper stops only the recorded application after checking command and working directory. Its negative test refused the isolated mock server; the mock was subsequently stopped explicitly.
- Controlled provider failure and generation exception produced safe UI messages and released admission. Empty prompt validation passed. Closing a browser during real GPU generation recorded `cancel / GeneratorExit`, released the slot in about five seconds and allowed a subsequent real queued generation and mesh download. Fault injection exists only in the opt-in profiling wrapper; it was disabled for the final service.
- Clean restart observed Warming / HTTP 503 / disabled Generate before Ready. The first final-process warmup took **176.40 s**. Five real mobile requests then passed latency and exact output checks, but handler-end RSS rose from **4.87 to 5.82 GiB**. GPU allocations settled and remained stable. This first soak is retained under `soak/`; it was not accepted as a complete host-memory stability result.

Memory refinement hypothesis: the old unload policy invoked `gc.collect()` between stages; permanent residency removes those calls, allowing CPU geometry reference cycles to persist. Added request-completion collection without unloading models or emptying CUDA caches. Focused lifecycle/Quick Trial/cleanup checks pass: **24 passed, 4 deselected**. An unreachable-cycle test validates reclamation. The revised real soak completed under `cleanup-soak/`, with before/after RSS and collection counts recorded in `cleanup-service/stages.jsonl`.

Public/phone validation and the stable QR redirect remain operator actions documented in `MANUAL_ACTION_REQUIRED.md`. Final results and the full operator runbook are consolidated in `../ECCV_READINESS_REPORT.md`.

## Final soak results

The revised lifecycle completed **ten sequential real requests** across `cleanup-soak/` and `cleanup-soak-extension/`. Full mean **45.666 s**, p50 **45.056 s**, worst **48.151 s**; backend mean **38.808 s**, planner p50 **2.050 s**. All ten GLBs are byte-identical; a loaded mesh has finite vertices and nonempty faces. Large Qwen loads after Ready: **zero**. Peak allocated GPU0/GPU1 **61.206/57.149 GiB**, reserved **62.658/58.707 GiB**; residency is exactly constant from request two through ten.

Collection averaged **0.370 s** and reduced the earlier continuous RSS rise. Post-collection RSS after the first five requests was 4.19, 4.58, 4.65, 4.71, 4.69 GiB. The next five fluctuated 4.85, 4.81, 5.19, 4.87, 5.05 GiB. This bounds observed memory during the short soak; it does not establish a long-duration no-leak guarantee. Keep watching host RSS during the session. Final startup took **308.11 s**, with Warming/disabled Generate observed before Ready.

Final core regression: **274 passed, 27 deselected**. Ruff, shell syntax and whitespace checks passed. Credential-value scan found zero matches; Gradio denied private credential-file access with HTTP 403 without recording the response body. Mobile screenshot confirms no horizontal overflow, prominent Generate, visible Ready and usable final viewer/download. Actual phone/cellular and proxy generation remain unverified.

The first customization harness acted before the final UI update settled and therefore performed no actual edit; its apparent success was rejected during stage-timing inspection. The corrected harness waits for the settled form and verifies the new label. Real Braille edit then passed in **27.629 s**, invoking only mesh/geometry stages, with no planner or image regeneration (`edit-check-v2/result.json`). Current application source matches its startup checksums; fallback checkout has no tracked modifications. Final health is Ready with an empty queue; the service remains running and its dedicated measurement trace has been stopped.
