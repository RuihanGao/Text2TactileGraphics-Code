# Stage 3 — PASS (local traffic; external proxy remains pending)

Hypothesis: admission before planning, a whole-workflow lock and two pending slots bound wait and prevent adapter races without adding GPU concurrency.

The first six-client lightweight test exposed missing persistent rejection feedback: three jobs completed, three correctly rejected jobs looked like client timeouts. Added a cookie-scoped visible busy notice and repeated the test: **3 success / 3 friendly rejection / 0 errors**. Cookie-scoped duplicates across Gradio sessions, disabled Generate, refresh/re-submit rejection, disconnected-job cleanup and successful later submission all passed against the isolated lightweight server. It is bound only to localhost:8083 and is never used by the production launcher.

Actual GPU escalation used the real planner and pipeline, only after lightweight guards passed:

| Virtual users | Completed | Rejected | Errors | Browser max wait s | Observed successful requests/hour |
|---|---:|---:|---:|---:|---:|
| 1 | 1 | 0 | 0 | 3.28 | 66.66 |
| 2 | 2 | 0 | 0 | 49.07 | 70.53 |
| 4 | 3 | 1 | 0 | 94.31 | 75.44 |
| 6 | 3 | 3 | 0 | 93.77 | 75.41 |

All **9 accepted requests succeeded**; 4 overload rejections. Telemetry proves maximum **1 active workflow**, maximum server queue wait **92.16 s**, and mean service duration **46.74 s**. All nine downloaded GLBs are byte-identical. No CUDA OOM, accepted-job failure, unexpected HTTP failure or adapter contamination occurred; no large Qwen loads after Ready.

## Recommended poster configuration

- `TEXT2TACTILEGRAPHICS_POSTER_MODE=1`
- `TEXT2TACTILEGRAPHICS_GPU_CONCURRENCY=1` (other values rejected)
- `TEXT2TACTILEGRAPHICS_MAX_PENDING=2`
- `TEXT2TACTILEGRAPHICS_BUSY_WAIT_LIMIT_SECONDS=180`
- Expected service: roughly 45–50 s. Position one: about one minute; position two: about 1–2 minutes. Observed saturated bursts: about **75 requests/hour**. These are measured local-client observations, not guarantees for arbitrary eight-region/40-step requests.

Admission uses the recent observed service durations conservatively to reject projected waits beyond budget. Expired pending tickets are invalidated before planner/GPU work; UI gets a retry notice. A separate scheduled slot distinguishes an accepted next-to-run job from pending jobs. Server telemetry retains random request IDs, acceptance/rejection, queue depth, start/finish/wait/duration, success/failure/cancel and underlying error class, without prompt contents. Browser cookies prevent accidental duplicate jobs across refreshes; they are not authentication. Direct expensive HTTP execution stays disabled and queued calls require a valid owner-bound admission ticket.

Peak allocated GPU0/GPU1: **61.206/57.158 GiB**. Pure CPU geometry spans total about **25.77 s/request**, primarily mesh construction and displacement. Detailed geometry/allocator and 1 Hz utilization/NVML data are in `aggregate.json`, per-burst JSON/CSV/GPU traces and `service/stages.jsonl`. Low aggregate GPU utilization reflects sequential inference and CPU geometry while weights remain resident.

Focused admission/Quick Trial/health/lifecycle tests passed (25 before final observability refinements; final admission check saved separately). Public-proxy load testing remains blocked by external exposure/validation from Stage 2; no external traffic success is claimed. `scripts/load_test_quick_trial.py` supports a public or local URL, user count, burst/staggered arrivals and CSV/JSON results.
