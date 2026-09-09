"""Browser load test with fixed safe prompts, bounded bursts and JSON/CSV output."""

import argparse
import asyncio
import csv
import json
import shutil
import subprocess
import time
from pathlib import Path

from playwright.async_api import async_playwright

PROMPT = "a dolphin with wings with an avocado skin texture."


async def run(args):
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    trace = None
    if shutil.which("nvidia-smi"):
        trace_file = (out / "gpu.csv").open("w")
        trace = subprocess.Popen(
            [
                "nvidia-smi",
                "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw",
                "--format=csv,nounits",
                "-l",
                "1",
            ],
            stdout=trace_file,
        )
    rows = []
    try:
        async with async_playwright() as pw:
            browser = await pw.chromium.launch(headless=True, args=["--no-sandbox"])
            barrier = asyncio.Event()
            arrived = 0

            async def user(index):
                nonlocal arrived
                context = await browser.new_context(
                    viewport={"width": 390, "height": 844},
                    is_mobile=True,
                    has_touch=True,
                )
                page = await context.new_page()
                row = {
                    "user": index,
                    "accepted": False,
                    "outcome": "error",
                    "wait_seconds": None,
                    "seconds": None,
                    "http_failures": 0,
                }
                failures = []
                page.on(
                    "response",
                    lambda r: failures.append(r.status) if r.status >= 400 else None,
                )
                try:
                    await page.goto(
                        args.url, wait_until="domcontentloaded", timeout=60000
                    )
                    await page.get_by_role(
                        "button", name="Generate", exact=True
                    ).wait_for(timeout=60000)
                    await page.get_by_label(
                        "What would you like to create?", exact=True
                    ).fill(PROMPT)
                    arrived += 1
                    if arrived == args.users:
                        barrier.set()
                    await asyncio.wait_for(barrier.wait(), timeout=90)
                    await asyncio.sleep(index * args.arrival_gap)
                    start = time.monotonic()
                    row["started"] = time.time()
                    await page.get_by_role("button", name="Generate", exact=True).click(
                        timeout=10000
                    )
                    events = []
                    previous = None
                    while time.monotonic() - start < args.timeout:
                        status = await page.locator("#quick-status").inner_text()
                        body = await page.locator("body").inner_text()
                        if (
                            "The live demo is busy right now." in body
                            or "You already have a request" in body
                        ):
                            row["outcome"] = "rejected"
                            break
                        if previous != status:
                            events.append(
                                {"status": status, "seconds": time.monotonic() - start}
                            )
                            previous = status
                        if (
                            status in ("Understanding prompt", "Generating object")
                            and row["wait_seconds"] is None
                        ):
                            row["wait_seconds"] = time.monotonic() - start
                            row["accepted"] = True
                        if "Could not finish:" in status:
                            row["outcome"] = "generation_error"
                            break
                        if status == "Done":
                            row["accepted"] = True
                            row["outcome"] = "success"
                            row["seconds"] = time.monotonic() - start
                            async with page.expect_download(timeout=30000) as transfer:
                                await page.get_by_text(
                                    "Download mesh", exact=True
                                ).click()
                            download = await transfer.value
                            await download.save_as(str(out / f"user-{index}.glb"))
                            assert (out / f"user-{index}.glb").stat().st_size > 1000
                            break
                        await asyncio.sleep(0.15)
                    else:
                        row["outcome"] = "timeout"
                    row["seconds"] = row["seconds"] or time.monotonic() - start
                    row["events"] = events
                    # Gradio's friendly admission rejection may use an HTTP error;
                    # distinguish it from unexpected transport failures.
                    row["http_failures"] = (
                        0 if row["outcome"] == "rejected" else len(failures)
                    )
                    row["observed_http_error_statuses"] = failures
                    await page.screenshot(
                        path=str(out / f"user-{index}.png"), full_page=True
                    )
                except Exception as exc:
                    row["error_class"] = type(exc).__name__
                    barrier.set()
                finally:
                    rows.append(row)
                    (out / "results.json").write_text(json.dumps(rows, indent=2))
                    await context.close()

            began = time.monotonic()
            await asyncio.gather(*(user(i) for i in range(args.users)))
            elapsed = time.monotonic() - began
            success = sum(r["outcome"] == "success" for r in rows)
            summary = {
                "users": args.users,
                "accepted": sum(r["accepted"] for r in rows),
                "success": success,
                "rejected": sum(r["outcome"] == "rejected" for r in rows),
                "errors": sum(
                    r["outcome"] not in ("success", "rejected") for r in rows
                ),
                "wall_seconds": elapsed,
                "observed_requests_per_hour": success * 3600 / elapsed,
                "max_wait_seconds": max((r["wait_seconds"] or 0) for r in rows),
                "url": args.url,
            }
            (out / "summary.json").write_text(json.dumps(summary, indent=2))
            with (out / "results.csv").open("w") as f:
                fields = [
                    "user",
                    "accepted",
                    "outcome",
                    "wait_seconds",
                    "seconds",
                    "http_failures",
                ]
                writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
                writer.writeheader()
                writer.writerows(rows)
            print(json.dumps(summary, indent=2))
            await browser.close()
            return summary["errors"] == 0
    finally:
        if trace:
            trace.terminate()
            trace.wait()
            trace_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8080")
    parser.add_argument("--users", type=int, default=1)
    parser.add_argument(
        "--arrival-gap",
        type=float,
        default=0,
        help="Seconds between arrivals; zero is a simultaneous burst",
    )
    parser.add_argument("--timeout", type=float, default=240)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if not 1 <= args.users <= 20:
        parser.error("Use 1–20 users; escalate real GPU tests gradually")
    raise SystemExit(0 if asyncio.run(run(args)) else 1)
