"""Verify cached Qwen weight bytes against recorded Hugging Face LFS identities."""

import hashlib
import json
from pathlib import Path

root = Path("profiling/results/single-gpu-quality")
manifest = json.loads((root / "checkpoint-identities.json").read_text())
records = []
for row in manifest["qwen_files"]:
    path = Path(row["file"])
    if path.suffix != ".safetensors":
        continue
    before = path.stat()
    with path.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    after = path.stat()
    assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns)
    result = dict(
        **row,
        actual_sha256=sha,
        sha256_matches_lfs_etag=sha == row["etag"],
        mtime_ns=after.st_mtime_ns,
    )
    records.append(result)
    (root / "verified-qwen-weights.json").write_text(json.dumps(records, indent=2))
    print(path.name, result["sha256_matches_lfs_etag"], flush=True)
assert all(r["sha256_matches_lfs_etag"] for r in records)
