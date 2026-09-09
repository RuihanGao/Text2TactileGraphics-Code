"""Check cookie-scoped admission across Gradio sessions without GPU inference."""

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

with sync_playwright() as p:
    browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
    page = browser.new_page()
    page.goto("http://127.0.0.1:8083")
    page.get_by_label("What would you like to create?", exact=True).fill("a dolphin")
    page.get_by_role("button", name="Generate", exact=True).click()
    page.wait_for_function(
        "document.querySelector('#quick-status').textContent.includes('Generating object')"
    )
    config = page.request.get("http://127.0.0.1:8083/config").json()
    fn = next(d for d in config["dependencies"] if d.get("api_name") == "reserve")
    response = page.request.post(
        "http://127.0.0.1:8083/gradio_api/run/reserve",
        data={
            "data": [],
            "fn_index": fn["id"],
            "session_hash": "different-gradio-session",
        },
    )
    payload = response.json()
    assert "already have a request" in str(payload)
    page.wait_for_function(
        "document.body.textContent.includes('You already have a request')",
        timeout=10000,
    )
    assert page.get_by_role("button", name="Generate", exact=True).is_disabled()
    page.wait_for_function(
        "document.querySelector('#quick-status').textContent.trim()==='Done'",
        timeout=15000,
    )
    Path("profiling/results/eccv/stage3-traffic/duplicate-check.json").write_text(
        json.dumps(
            {
                "passed": True,
                "different_session_same_cookie_rejected": True,
                "generate_disabled": True,
                "first_job_completed": True,
            },
            indent=2,
        )
    )
    browser.close()
