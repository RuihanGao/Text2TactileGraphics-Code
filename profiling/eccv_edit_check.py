"""Exercise the real mobile customization binding after normal generation."""

import argparse
import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

p = argparse.ArgumentParser()
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
a.output.mkdir(parents=True, exist_ok=True)
with sync_playwright() as pw:
    browser = pw.chromium.launch(args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 390, "height": 844})
    page.goto("http://127.0.0.1:8080")
    page.get_by_label("What would you like to create?", exact=True).fill(
        "a dolphin with wings with an avocado skin texture."
    )
    page.get_by_role("button", name="Generate", exact=True).click()
    page.wait_for_function(
        "document.querySelector('#quick-status')?.innerText === 'Done'", timeout=180000
    )
    page.wait_for_timeout(3000)
    page.get_by_text("Customize intermediate steps", exact=True).click()
    page.get_by_label("Braille label", exact=True).fill("ECCV")
    page.get_by_label("Braille label", exact=True).press("Tab")
    page.wait_for_timeout(500)
    assert page.get_by_label("Braille label", exact=True).input_value() == "ECCV"
    started = time.time()
    page.get_by_role("button", name="Apply edits and continue", exact=True).click()
    page.wait_for_function(
        "document.querySelector('#quick-status')?.innerText !== 'Done'", timeout=30000
    )
    page.wait_for_function(
        "document.querySelector('#quick-status')?.innerText === 'Done'", timeout=180000
    )
    finished = time.time()
    assert page.get_by_label("Braille label", exact=True).input_value() == "ECCV"
    with page.expect_download() as transfer:
        page.get_by_text("Download mesh", exact=True).click()
    transfer.value.save_as(str(a.output / "edited.glb"))
    assert (a.output / "edited.glb").stat().st_size > 10000
    (a.output / "result.json").write_text(
        json.dumps(
            {
                "passed": True,
                "edit_started": started,
                "edit_finished": finished,
                "edit_seconds": finished - started,
            },
            indent=2,
        )
    )
    browser.close()
