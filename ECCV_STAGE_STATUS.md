# ECCV stage status — historical hardening snapshot

Current handoff (2026-09-09): the operator reports external laptop/phone validation
completed on the hardened service. The stages below describe the earlier test
session; external load testing and stable QR redirect validation are not inferred
from that confirmation. See [ECCV_DEMO.md](ECCV_DEMO.md) for current operation.

- Stage 0: PASS — frozen environment, two suitable GPUs, 264 tests passed.
- Stage 1: PASS — exact reference outputs; five full warm browser requests average 45.52 s.
- Stage 2: WAITING_FOR_EXTERNAL_VALIDATION — local real queue/download passes; public proxy returns 403/404; operator actions documented.
- Stage 3: PASS (local) — 1/2/4/6-user escalation, 9 successful jobs, 4 friendly rejections, no OOM; external load path pending.
- Stage 4: PASS (local) — ten warm requests average 45.67 s; stable GPU residency, bounded short-soak host memory, failure/restart recovery and real customization passed; full runbook saved. Physical-phone validation remains pending under Stage 2.

Overall: NOT READY FOR POSTER until the RunPod public proxy and an actual outside-client generation are validated. See `MANUAL_ACTION_REQUIRED.md` and `profiling/results/eccv/ECCV_READINESS_REPORT.md`.
