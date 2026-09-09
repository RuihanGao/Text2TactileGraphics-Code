"""Bounded admission before planning; one complete GPU workflow at a time."""

import gc
import json
import math
import os
import threading
import time
import uuid
from collections import deque
from contextlib import contextmanager
from pathlib import Path

BUSY = "The live demo is busy right now. A few people are already generating tactile graphics. Please try again in about 1–2 minutes."


class Admission:
    def __init__(
        self, max_pending=2, wait_limit=180, telemetry=None, initial_seconds=60
    ):
        if max_pending < 0 or wait_limit <= 0:
            raise ValueError(
                "Poster queue limits must be nonnegative with a positive wait budget"
            )
        self.max_pending = max_pending
        self.wait_limit = wait_limit
        self.telemetry = Path(telemetry) if telemetry else None
        self.durations = deque([initial_seconds], maxlen=20)
        self.jobs = {}
        self.notices = {}
        self.lock = threading.RLock()
        self.gpu_lock = threading.Lock()

    @classmethod
    def from_environment(cls):
        if int(os.getenv("TEXT2TACTILEGRAPHICS_GPU_CONCURRENCY", "1")) != 1:
            raise ValueError(
                "Poster GPU concurrency must remain one until separately validated"
            )
        return cls(
            int(os.getenv("TEXT2TACTILEGRAPHICS_MAX_PENDING", "2")),
            float(os.getenv("TEXT2TACTILEGRAPHICS_BUSY_WAIT_LIMIT_SECONDS", "180")),
            os.getenv(
                "TEXT2TACTILEGRAPHICS_TELEMETRY",
                "/tmp/text2tactilegraphics-events.jsonl",
            ),
        )

    def emit(self, event, **data):
        if self.telemetry:
            self.telemetry.parent.mkdir(parents=True, exist_ok=True)
            with self.telemetry.open("a") as stream:
                stream.write(
                    json.dumps({"timestamp": time.time(), "event": event, **data})
                    + "\n"
                )

    def expire(self):
        now = time.monotonic()
        for token, job in list(self.jobs.items()):
            if job["start"] is None and now - job["accepted"] > self.wait_limit:
                self.jobs.pop(token)
                self.notices[job["owner"]] = (
                    now,
                    "Your queued request reached the wait limit. Please try Generate again.",
                )
                self.emit(
                    "cancel",
                    request_id=token,
                    reason="wait_budget",
                    wait_seconds=now - job["accepted"],
                )

    def reserve(self, owner):
        with self.lock:
            self.expire()
            if not owner:
                raise ValueError("Please reload the demo before generating.")
            if any(j["owner"] == owner for j in self.jobs.values()):
                self.notices[owner] = (
                    time.monotonic(),
                    "You already have a request in progress. Please wait for it to finish.",
                )
                self.emit("rejected", reason="duplicate", queue_depth=len(self.jobs))
                raise ValueError(
                    "You already have a request in progress. Please wait for it to finish."
                )
            estimate = max(self.durations)
            if (
                len(self.jobs) >= self.max_pending + 1
                or len(self.jobs) * estimate > self.wait_limit
            ):
                self.notices[owner] = (time.monotonic(), BUSY)
                self.emit("rejected", reason="capacity", queue_depth=len(self.jobs))
                raise ValueError(BUSY)
            token = uuid.uuid4().hex
            self.notices.pop(owner, None)
            self.jobs[token] = {
                "owner": owner,
                "accepted": time.monotonic(),
                "start": None,
            }
            self.emit("accepted", request_id=token, queue_depth=len(self.jobs) - 1)
            return token

    def notice(self, owner):
        with self.lock:
            now = time.monotonic()
            self.notices = {k: v for k, v in self.notices.items() if now - v[0] < 120}
            while len(self.notices) > 1024:
                self.notices.pop(next(iter(self.notices)))
            return self.notices.get(owner, (0, ""))[1]

    def owns(self, owner):
        with self.lock:
            self.expire()
            return any(j["owner"] == owner for j in self.jobs.values())

    def snapshot(self, owner=None):
        with self.lock:
            self.expire()
            active = sum(j["start"] is not None for j in self.jobs.values())
            scheduled = int(bool(self.jobs) and not active)
            ahead = next(
                (i for i, j in enumerate(self.jobs.values()) if j["owner"] == owner),
                None,
            )
            return {
                "active": active,
                "pending": len(self.jobs) - active - scheduled,
                "scheduled": scheduled,
                "capacity": self.max_pending + 1,
                "jobs_ahead": ahead,
                "service_seconds": sum(self.durations) / len(self.durations),
                "wait_bucket_minutes": None
                if ahead is None
                else math.ceil(ahead * max(self.durations) / 60),
            }

    @contextmanager
    def run(self, token, owner):
        # This lock covers all stages and survives browser transport cancellation
        # until the executing synchronous generator releases it.
        with self.gpu_lock:
            with self.lock:
                self.expire()
                job = self.jobs.get(token)
                if job is None or job["owner"] != owner or job["start"] is not None:
                    raise ValueError(
                        "This request is no longer queued. Please try Generate again."
                    )
                job["start"] = time.monotonic()
                wait = job["start"] - job["accepted"]
                self.emit("service_start", request_id=token, wait_seconds=wait)
            outcome, error = "success", None
            try:
                yield
            except BaseException as exc:
                outcome = "cancel" if isinstance(exc, GeneratorExit) else "failure"
                root = exc
                seen = set()
                while id(root) not in seen:
                    seen.add(id(root))
                    cause = root.__cause__ or root.__context__
                    if cause is None:
                        break
                    root = cause
                error = type(root).__name__
                raise
            finally:
                # Geometry libraries can leave large CPU arrays in reference
                # cycles. The resident lifecycle no longer runs unload_model's
                # collection between stages; reclaim cycles without evicting
                # any model or touching the CUDA allocator cache.
                gc.collect()
                with self.lock:
                    duration = time.monotonic() - job["start"]
                    self.jobs.pop(token, None)
                    if outcome == "success":
                        self.durations.append(duration)
                    self.emit(
                        "service_finish",
                        request_id=token,
                        outcome=outcome,
                        error_class=error,
                        wait_seconds=wait,
                        execution_seconds=duration,
                    )


def request_owner(request):
    if request is None:
        return None
    underlying = getattr(request, "request", None)
    return (
        underlying.cookies.get("eccv_client") if underlying else None
    ) or request.session_hash
