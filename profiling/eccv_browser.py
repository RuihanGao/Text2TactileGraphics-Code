"""Sequential real mobile-browser latency and download checks."""

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

p = argparse.ArgumentParser()
p.add_argument("--url", default="http://127.0.0.1:8080")
p.add_argument("--output", type=Path, required=True)
p.add_argument("--runs", type=int, default=5)
args = p.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
rows = []
with sync_playwright() as pw:
    browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(
        viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True
    )
    errors = []
    page.on("pageerror", lambda e: errors.append(type(e).__name__))
    for attempt in range(90):
        try:
            page.goto(args.url, wait_until="domcontentloaded")
            break
        except Exception:
            if attempt == 89:
                raise
            time.sleep(1)
    page.get_by_role("button", name="Generate", exact=True).wait_for(timeout=90000)
    (args.output / "startup-observation.json").write_text(
        json.dumps(
            {
                "generate_disabled": page.get_by_role(
                    "button", name="Generate", exact=True
                ).is_disabled(),
                "warming_visible": "Service: Warming"
                in page.locator("body").inner_text(),
            },
            indent=2,
        )
    )
    page.screenshot(path=str(args.output / "startup-observation.png"), full_page=True)
    page.wait_for_function(
        "Array.from(document.querySelectorAll('button')).some(b=>b.textContent.trim()==='Generate'&&!b.disabled)",
        timeout=900000,
    )
    assert page.title() == "Text2TactileGraphics · Quick Trial"
    assert not page.get_by_label("Shape prompt", exact=True).is_visible()
    assert page.evaluate("document.documentElement.scrollWidth <= window.innerWidth")
    page.screenshot(path=str(args.output / "mobile-start.png"), full_page=True)
    for i in range(args.runs):
        folder = args.output / str(i)
        folder.mkdir(exist_ok=True)
        page.get_by_label("What would you like to create?", exact=True).fill(
            "a dolphin with wings with an avocado skin texture."
        )
        start = time.monotonic()
        wall = time.time()
        page.get_by_role("button", name="Generate", exact=True).click()
        events = []
        previous = None
        started = False
        while time.monotonic() - start < 600:
            status = page.locator("#quick-status").inner_text()
            if status != previous:
                events.append({"status": status, "seconds": time.monotonic() - start})
                previous = status
            if status != "Done":
                started = True
            if status.startswith("Could not finish:"):
                raise RuntimeError(status)
            if status == "Done" and started:
                break
            page.wait_for_timeout(100)
        else:
            raise TimeoutError("Generation did not complete within 600 seconds")
        latency = time.monotonic() - start
        with page.expect_download(timeout=30000) as transfer:
            page.get_by_text("Download mesh", exact=True).click()
        transfer.value.save_as(str(folder / "final.glb"))
        assert (folder / "final.glb").stat().st_size > 10000
        row = {
            "run": i,
            "started": wall,
            "seconds": latency,
            "events": events,
            "page_errors": list(errors),
        }
        rows.append(row)
        (args.output / "results.json").write_text(json.dumps(rows, indent=2))
        print(json.dumps(row), flush=True)
    page.wait_for_timeout(3000)
    page.screenshot(path=str(args.output / "mobile-done.png"), full_page=True)
    assert not errors
    browser.close()
