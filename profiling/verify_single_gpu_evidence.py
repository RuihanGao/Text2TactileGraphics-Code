"""Validate run counts, isolated residency, checkpoint consistency, and stability."""

import hashlib
import json
from pathlib import Path

root = Path("profiling/results/single-gpu-quality")
checks = {}
for name in ("equivalence-targeted", "equivalence-reuse"):
    rows = json.loads((root / f"{name}.json").read_text())
    assert len(rows) == 3
    assert all(
        all(
            r[k]
            for k in (
                "exact_file",
                "inference_config_equal",
                "component_dtype_device_adapters_equal",
                "scheduler_after_equal",
                "operations_equal",
            )
        )
        for r in rows
    )
    checks[name] = True
for stage, model in [
    ("base", "qwen_base_edit"),
    ("texture", "qwen_texture"),
    ("tiling", "qwen_tiling"),
]:
    rows = [
        json.loads(line)
        for line in (root / f"reference-{stage}/stages.jsonl").read_text().splitlines()
    ]
    loaded = {
        r["name"].removeprefix("load.") for r in rows if r["kind"] == "model_load"
    }
    assert loaded == {model}, loaded
    checks[f"isolated-{stage}-only-required-model"] = True


# File paths and all construction dtypes/devices must match; sharing deliberately
# maps the clean texture construction to tiling because these configs are equal.
def configs(directory):
    return [
        r["configs"]
        for r in map(json.loads, (directory / "runtime.jsonl").read_text().splitlines())
        if r["event"] == "resolved_checkpoints"
    ]


reference = [configs(root / f"reference-{s}")[0] for s in ["base", "texture", "tiling"]]
assert configs(root / "dolphin-validation") == reference
assert configs(root / "dolphin-reuse-validation") == reference[:2]
assert reference[1] == reference[2]
checks["resolved-checkpoints-and-loader-configs-equal"] = True
warm = []
for path in sorted((root / "dolphin-performance").glob("cold-*")):
    record = {}
    for run in ["cold", "warm1", "warm2", "warm3"]:
        candidate = path.with_name(run + "-" + path.name.removeprefix("cold-"))
        record[run] = hashlib.sha256(candidate.read_bytes()).hexdigest()
    assert len(set(record.values())) == 1
    warm.append(dict(artifact=path.name.removeprefix("cold-"), hashes=record))
checks["performance-all-artifacts-byte-identical"] = warm
rows = [
    json.loads(line)
    for line in (root / "dolphin-performance/stages.jsonl").read_text().splitlines()
]
totals = [r for r in rows if r["kind"] == "total"]
assert [r["run"] for r in totals] == ["cold", "warm1", "warm2", "warm3"]
assert all(r["status"] == "ok" for r in totals)
assert all(
    sum(r.get("name") == "qwen.inference" and r["run"] == run for r in rows) == 3
    for run in ["cold", "warm1", "warm2", "warm3"]
)
checks["cold-plus-three-warm-complete"] = True
for example, count in [("dolphin", 8), ("lamp", 14)]:
    d = json.loads((root / f"{example}-lifecycle-artifacts.json").read_text())
    assert len(d["artifacts"]) == count
    assert all(r["exact_file"] for r in d["artifacts"])
    assert all(all(v for k, v in r.items() if k != "call") for r in d["qwen_calls"])
    checks[f"{example}-full-lifecycle-equality"] = True
weights = json.loads((root / "verified-qwen-weights.json").read_text())
assert len(weights) == 19 and all(r["sha256_matches_lfs_etag"] for r in weights)
checks["19-qwen-weight-files-sha256-verified"] = True
(root / "verification.json").write_text(json.dumps(checks, indent=2))
print("All single-GPU evidence checks passed.")
