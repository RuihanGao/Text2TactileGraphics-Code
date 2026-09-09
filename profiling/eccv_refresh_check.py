"""Exercise refresh/disconnection and subsequent admission on the local fixture."""

import json
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page()
    url = "http://127.0.0.1:8083"
    page.goto(url)
    page.get_by_label("What would you like to create?", exact=True).fill("a dolphin")
    page.get_by_role("button", name="Generate", exact=True).click()
    page.wait_for_function(
        "document.querySelector('#quick-status').textContent.includes('Generating object')"
    )
    cookie = next(
        c["value"] for c in page.context.cookies() if c["name"] == "eccv_client"
    )
    page.reload()
    assert cookie == next(
        c["value"] for c in page.context.cookies() if c["name"] == "eccv_client"
    )
    config = page.request.get(url + "/config").json()
    fn = next(d for d in config["dependencies"] if d.get("api_name") == "reserve")
    response = page.request.post(
        url + "/gradio_api/run/reserve",
        data={"data": [], "fn_index": fn["id"], "session_hash": "refreshed-session"},
    )
    assert "already have a request" in str(response.json())
    for _ in range(20):
        queue = page.request.get(url + "/healthz").json()["queue"]
        if not queue["active"] and not queue["pending"]:
            break
        time.sleep(0.5)
    else:
        raise AssertionError("Disconnected job did not release admission")
    page.get_by_label("What would you like to create?", exact=True).fill("a dolphin")
    page.get_by_role("button", name="Generate", exact=True).click(timeout=10000)
    page.wait_for_function(
        "document.querySelector('#quick-status').textContent.trim()==='Done'",
        timeout=20000,
    )
    Path("profiling/results/eccv/stage3-traffic/refresh-check.json").write_text(
        json.dumps(
            {
                "passed": True,
                "cookie_retained": True,
                "refresh_duplicate_rejected": True,
                "disconnected_job_released": True,
                "later_job_completed": True,
            },
            indent=2,
        )
    )
    browser.close()
