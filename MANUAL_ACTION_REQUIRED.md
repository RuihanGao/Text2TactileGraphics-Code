# External deployment verification checklist

Current handoff (2026-09-09): the operator reports the hardened demo working externally on laptop and phone. The earlier same-Pod 403 probe is historical. Use this checklist after a URL/deployment change; packaging did not repeat GPU/browser validation. A stable QR redirect check is separate.

1. Confirm the RunPod console exposes internal **HTTP 8080**. Do not reconfigure an already-working Pod: changes can reset container-local tooling. The application binds `0.0.0.0:8080`; nginx 8081 is unnecessary. Use the console's HTTP connection for 8080 and compare it with `https://<POD_ID>-8080.proxy.runpod.net`.
2. Wait until `https://<POD_ID>-8080.proxy.runpod.net/readyz` returns HTTP 200 with `ready: true`. HTTP 503 means models are still warming or unhealthy; inspect local logs before proceeding.
3. On your laptop **outside RunPod**, from this repository checkout, run:

```bash
python -m pip install playwright
python -m playwright install chromium
python scripts/verify_public_demo.py --url "https://<POD_ID>-8080.proxy.runpod.net" --external --output external-validation.json
```

Save/share `external-validation.json` and its screenshot. The command submits a real queued request and downloads the final mesh. A same-Pod invocation is only a proxy-path test, even with `--external`.

4. On your phone, disable Wi-Fi and open `https://<POD_ID>-8080.proxy.runpod.net` on cellular. Confirm Quick Trial, visible Ready, Generate, progressive status, final 3D viewer and Download mesh. Complete at least one generation; record success/failure, UTC time, browser/network and screenshot. No password is required.
5. After these checks pass, update the existing stable QR redirect destination to `https://<POD_ID>-8080.proxy.runpod.net` and scan the printed QR from the phone again. The redirect provider/account is not available inside this Pod; no change has been claimed.

Do not mark external access or poster readiness PASS until the actual outside-client result is recorded.
