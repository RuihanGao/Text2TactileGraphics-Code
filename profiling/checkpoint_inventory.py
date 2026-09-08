"""Record cached immutable checkpoint identities without accessing credentials."""

import hashlib
import json
from pathlib import Path

root = Path("/workspace/model_cache/diffsynth")
rows = []
for metadata in sorted(root.glob("Qwen/*/.cache/huggingface/download/**/*.metadata")):
    fields = metadata.read_text().splitlines()
    base = metadata.parents[
        next(i for i, p in enumerate(metadata.parents) if p.name == ".cache")
    ].parent
    relative = str(metadata).split("/download/", 1)[1].removesuffix(".metadata")
    target = base / relative
    rows.append(
        dict(
            file=str(target),
            bytes=target.stat().st_size if target.exists() else None,
            revision=fields[0],
            etag=fields[1],
        )
    )
refs = {
    str(p.parent.parent.name): p.read_text().strip()
    for p in Path("/workspace/model_cache/huggingface/hub").glob("models--*/refs/main")
}
custom = []
for p in sorted(
    Path("/workspace/model_cache/text2tactilegraphics/ckpt").glob("*.safetensors")
):
    with p.open("rb") as f:
        sha = hashlib.file_digest(f, "sha256").hexdigest()
    custom.append(dict(file=str(p), bytes=p.stat().st_size, sha256=sha))
Path("profiling/results/single-gpu-quality/checkpoint-identities.json").write_text(
    json.dumps(dict(qwen_files=rows, hub_revisions=refs, custom=custom), indent=2)
)
print(len(rows), "Qwen file identities;", len(custom), "custom checkpoints hashed")
