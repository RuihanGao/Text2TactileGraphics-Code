"""Verify the mobile UI and real queued generation from a chosen client location.

Install: python -m pip install playwright; python -m playwright install chromium
"""

import argparse
import json
import os
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

PROMPT = "a dolphin with wings with an avocado skin texture."


def verify(url, output, timeout=600, external=False):
    result = {
        "url": url,
        "vantage": "external client"
        if external and not os.getenv("RUNPOD_POD_ID")
        else "local or same-Pod proxy-path",
        "passed": False,
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        result["phase"] = "page and static assets"
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=True, args=["--no-sandbox"])
            page = browser.new_page(
                viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True
            )
            assets = []
            transports = []
            errors = []

            def response(r):
                if "/assets/" in r.url:
                    assets.append(r.status)
                if "/queue/data" in r.url and "text/event-stream" in r.headers.get(
                    "content-type", ""
                ):
                    transports.append(r.status)

            page.on("response", response)
            page.on("pageerror", lambda e: errors.append(type(e).__name__))
            response_page = page.goto(url, wait_until="domcontentloaded", timeout=30000)
            result["page_status"] = response_page.status
            assert response_page.ok, "Public page failed"
            page.get_by_role("button", name="Generate", exact=True).wait_for(
                timeout=60000
            )
            assert "Quick Trial" in page.title(), "Wrong application title"
            assert assets and all(s == 200 for s in assets), "Static assets failed"
            result["static_assets"] = len(assets)
            result["phase"] = "readiness"
            health = page.request.get(url.rstrip("/") + "/readyz", timeout=10000)
            result["readiness_status"] = health.status
            result["health"] = health.json()
            assert health.ok and result["health"]["ready"], "Service is not Ready"
            assert not page.get_by_label("Shape prompt", exact=True).is_visible(), (
                "Advanced must start collapsed"
            )
            assert page.evaluate(
                "document.documentElement.scrollWidth <= window.innerWidth"
            ), "Mobile overflow"
            page.get_by_label("What would you like to create?", exact=True).fill(PROMPT)
            result["phase"] = "queued generation"
            start = time.monotonic()
            page.get_by_role("button", name="Generate", exact=True).click()
            events = []
            previous = None
            while time.monotonic() - start < timeout:
                status = page.locator("#quick-status").inner_text()
                if (
                    "The live demo is busy right now."
                    in page.locator("body").inner_text()
                ):
                    result["admission_rejected"] = True
                    raise RuntimeError("Admission capacity exhausted")
                if previous != status:
                    events.append(
                        {"seconds": time.monotonic() - start, "status": status}
                    )
                    previous = status
                if "Could not finish:" in status:
                    raise RuntimeError("Generation reported a failure")
                if status == "Done":
                    break
                page.wait_for_timeout(200)
            else:
                raise TimeoutError("Queued generation exceeded the configured deadline")
            result["seconds"] = time.monotonic() - start
            result["events"] = events
            result["queue_stream_http_statuses"] = transports
            assert transports and all(s == 200 for s in transports), (
                "Queue stream not observed"
            )
            result["phase"] = "mesh download"
            with page.expect_download(timeout=30000) as transfer:
                page.get_by_text("Download mesh", exact=True).click()
            assert transfer.value.failure() is None, "Mesh download failed"
            mesh = output.with_suffix(".glb")
            transfer.value.save_as(str(mesh))
            result["mesh_bytes"] = mesh.stat().st_size
            assert result["mesh_bytes"] > 10000, "Mesh download is empty"
            page.wait_for_timeout(3000)
            page.screenshot(path=str(output.with_suffix(".png")), full_page=True)
            result["page_errors"] = errors
            assert not errors, "Browser errors occurred"
            result["passed"] = True
            result["phase"] = "complete"
            browser.close()
    except Exception as exc:
        result["error_class"] = type(exc).__name__
        # Avoid arbitrary HTTP/provider diagnostics containing credentials.
        result["error"] = (
            f"Verification failed during {result.get('phase', 'startup')}; inspect the status fields, URL and deadline."
        )
    finally:
        output.write_text(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--output", default="public-demo-verification.json")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument(
        "--external",
        action="store_true",
        help="Run on a laptop outside RunPod; this is not sufficient when RUNPOD_POD_ID is present",
    )
    args = parser.parse_args()
    result = verify(args.url, args.output, args.timeout, args.external)
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["passed"] else 1)
