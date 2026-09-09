"""Stop only the verified ECCV application process recorded by the launcher."""

import argparse
import os
import signal
import time
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--pid-file", default="/tmp/text2tactilegraphics-eccv.pid")
args = parser.parse_args()
try:
    pid = int(Path(args.pid_file).read_text().strip())
except (OSError, ValueError):
    raise SystemExit(
        "No valid recorded demo PID; inspect the listener before stopping anything."
    )
proc = Path(f"/proc/{pid}")
if not proc.exists():
    print("Recorded demo process has already exited.")
    raise SystemExit(0)
expected = Path(__file__).resolve().parents[1]
command = (proc / "cmdline").read_bytes().split(b"\0")
if (proc / "cwd").resolve() != expected or not (
    b"text2tactilegraphics.ui.public_app" in command
    or b"profiling/eccv_serve.py" in command
):
    raise SystemExit("Refusing: recorded PID is not this checkout's ECCV application.")
os.kill(pid, signal.SIGTERM)
for _ in range(300):
    try:
        exited = "State:\tZ" in (proc / "status").read_text()
    except FileNotFoundError:
        exited = True
    if exited:
        print("Identified demo process stopped.")
        break
    time.sleep(0.1)
else:
    raise SystemExit(
        "Demo is still draining requests. Inspect its log; no forced kill was sent."
    )
