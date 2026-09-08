"""Run isolated references serially, each in a fresh Python process."""

import subprocess
import sys
from pathlib import Path

root = Path("profiling/results/single-gpu-quality")
for stage in ("base", "texture", "tiling"):
    cmd = [
        sys.executable,
        "-u",
        "profiling/validate_single_gpu.py",
        "--audit",
        "--stage",
        stage,
        "--output",
        str(root / f"reference-{stage}"),
    ]
    if stage == "tiling":
        cmd += ["--input", str(root / "dolphin-validation/cold-region1-normal.png")]
    with (root / f"reference-{stage}.log").open("w") as f:
        subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=True)
    print("Completed isolated 80gb", stage, flush=True)
subprocess.run(
    [
        sys.executable,
        "profiling/compare_single_gpu.py",
        "--candidate",
        str(root / "dolphin-validation"),
        "--references",
        str(root),
        "--output",
        str(root / "equivalence-targeted.json"),
    ],
    check=True,
)
