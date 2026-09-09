# Stage 2 — WAITING_FOR_EXTERNAL_VALIDATION

Hypothesis: mounting lightweight readiness diagnostics alongside the supported Gradio queue provides observable startup and long-running progress, while disabling direct queued-handler POSTs prevents API bypass.

Implemented `/healthz` (process alive; HTTP 200) and `/readyz` (HTTP 503 while Warming/Unhealthy, 200 only Ready), with GPU count, lifecycle, ready state and pending queue count. These endpoints do not load models or return credentials/filesystem paths. The application binds `0.0.0.0:8080`; proxy address is derived from `RUNPOD_POD_ID`. No nginx dependency or password login was added. `queue(api_open=False)` blocks direct HTTP execution; the actual POST test received 404 with an instruction to join the queue.

Focused tests: **13 passed, 4 deselected**. Local real mobile verification passed: page HTTP 200, **89 static assets**, Ready HTTP 200, queued SSE HTTP 200, progressive status through Done, **48.94 s** including browser click, and an **18,851,936-byte GLB download**, with zero page errors. Advanced starts collapsed and no horizontal overflow was observed. This restarted process took **314.22 s** to warm, versus 103.22 s previously: populated network-volume/page-cache load times vary materially, so operators must wait for Ready rather than assume a fixed startup delay.

Public proxy results are **not passing**: Python HTTP probe returned 403; Chromium verifier returned 404. Both ran inside the Pod and are only proxy-path checks. No external phone/laptop client has completed a request. Actual public-proxy long-queue behavior cannot be established until port exposure works. Local SSE success does not establish that proxy transport is reliable. No speculative transport replacement was made.

`scripts/verify_public_demo.py` accepts a base URL, tests page/title/static assets/readiness, follows a real queued generation to Done, downloads the mesh and preserves JSON/screenshot evidence. It distinguishes same-Pod from operator-declared outside execution. Exact RunPod exposure, laptop command, phone/cellular test and QR redirect actions are in `MANUAL_ACTION_REQUIRED.md`.

Evidence: `warming-health.json`, `api-bypass.json`, `local-verification.json` plus screenshot/GLB, `initial-proxy.json`, `proxy-verification.json`, focused test log and real service stage timings. Continue local traffic/recovery work while external action is pending.
