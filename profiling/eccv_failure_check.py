"""Controlled failures in the profiling wrapper, then real recovery and download."""

import json
import sys
import time
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from verify_public_demo import verify

root = Path("profiling/results/eccv/stage4-readiness")
faults = root / "test-service"
url = "http://127.0.0.1:8080"
result = {}


def health():
    return json.load(urllib.request.urlopen(url + "/healthz", timeout=10))


for _ in range(900):
    try:
        if health()["ready"]:
            break
    except Exception:
        pass
    time.sleep(1)
else:
    raise SystemExit("Readiness timeout")

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page(viewport={"width": 390, "height": 844}, is_mobile=True)
    page.goto(url)

    def fail(label, prompt, needle, fault=None):
        if fault:
            (faults / f"fault-{fault}").touch()
        page.get_by_label("What would you like to create?", exact=True).fill(prompt)
        page.get_by_role("button", name="Generate", exact=True).click(timeout=10000)
        page.wait_for_function(
            "([needle]) => {const s=document.querySelector('#quick-status').textContent; return s.includes('Could not finish:') && s.includes(needle)}",
            arg=[needle],
            timeout=30000,
        )
        current = health()
        assert current["ready"] and current["queue"]["active"] == 0
        result[label] = {
            "passed": True,
            "ready_after_failure": True,
            "ui_status": page.locator("#quick-status").inner_text(),
        }
        (root / "failure-check.json").write_text(json.dumps(result, indent=2))
        print(label + " passed", flush=True)

    fail(
        "provider_error",
        "a dolphin with wings with an avocado skin texture.",
        "temporarily unavailable",
        "plan",
    )
    fail(
        "generation_exception",
        "a dolphin with wings with an avocado skin texture.",
        "Generating object",
        "generate_base_image",
    )
    fail("invalid_prompt", "", "between 1 and 2000")
    page.get_by_label("What would you like to create?", exact=True).fill(
        "a dolphin with wings with an avocado skin texture."
    )
    page.get_by_role("button", name="Generate", exact=True).click(timeout=10000)
    page.wait_for_function(
        "document.querySelector('#quick-status').textContent.includes('Generating object')",
        timeout=30000,
    )
    page.wait_for_timeout(1500)
    page.close()
    began = time.monotonic()
    for _ in range(180):
        h = health()
        if (
            not h["queue"]["active"]
            and not h["queue"]["pending"]
            and not h["queue"].get("scheduled")
        ):
            break
        time.sleep(1)
    else:
        raise AssertionError("Disconnected job blocked later visitors")
    result["real_browser_disconnect"] = {
        "passed": True,
        "cleanup_seconds": time.monotonic() - began,
        "ready": h["ready"],
    }
    (root / "failure-check.json").write_text(json.dumps(result, indent=2))
    browser.close()

recovery = verify(url, root / "recovery-verification.json")
assert recovery["passed"]
result["later_real_request"] = {
    "passed": True,
    "seconds": recovery["seconds"],
    "mesh_bytes": recovery["mesh_bytes"],
}
(root / "failure-check.json").write_text(json.dumps(result, indent=2))
print("All controlled failures and real recovery passed", flush=True)
