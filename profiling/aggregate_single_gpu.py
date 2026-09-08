"""Aggregate synchronized, non-overlapping latency attribution and image stability."""

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path("profiling/results/single-gpu-quality")
reports = {}
geometry_names = {
    "mesh_construction",
    "mesh_conversion",
    "texture_integration",
    "plate_closure",
    "braille",
    "export",
}
for path in sorted(ROOT.glob("*/stages.jsonl")):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if not any(r.get("kind") == "total" for r in rows):
        continue  # An independently running experiment has not completed yet.
    runs = {}
    for total in [r for r in rows if r.get("kind") == "total"]:
        rname = total["run"]
        timed = [r for r in rows if r["run"] == rname and "seconds" in r]
        by_kind = {}
        loads = {}
        for r in timed:
            assert r["exclusive_seconds"] >= -0.001
            by_kind[r["kind"]] = by_kind.get(r["kind"], 0) + r["exclusive_seconds"]
            if r["kind"] == "model_load":
                loads[r["name"]] = loads.get(r["name"], 0) + r["seconds"]
        assert abs(sum(by_kind.values()) - total["seconds"]) < 0.001
        runs[rname] = dict(
            e2e_s=total["seconds"],
            status=total["status"],
            exclusive_by_kind_s=by_kind,
            model_load_inclusive_s=loads,
            qwen_loading_s=sum(v for k, v in loads.items() if "qwen" in k),
            geometry_exclusive_s=sum(
                r["exclusive_seconds"] for r in timed if r["name"] in geometry_names
            ),
            qwen_inference_s=sum(
                r["seconds"] for r in timed if r["name"] == "qwen.inference"
            ),
            peak_allocated_gib=max(r["after_gib"]["peak_allocated"] for r in timed),
            peak_reserved_gib=max(r["after_gib"]["peak_reserved"] for r in timed),
            peak_rss_gib=max(r["process_peak_rss_gib"] for r in timed),
        )
    gpu = list(csv.DictReader((path.parent / "gpu.csv").open()))

    def col(prefix):
        key = next(k for k in gpu[0] if k.strip().startswith(prefix))
        return [float(r[key]) for r in gpu]

    warm = [
        r["e2e_s"]
        for k, r in runs.items()
        if k.startswith("warm") and r["status"] == "ok"
    ]
    reports[path.parent.name] = dict(
        runs=runs,
        warm_mean_s=statistics.mean(warm) if warm else None,
        warm_min_s=min(warm) if warm else None,
        warm_max_s=max(warm) if warm else None,
        nvml_peak_gib=max(col("memory.used")) / 1024,
        gpu_util_mean=statistics.mean(col("utilization.gpu")),
        memory_util_mean=statistics.mean(col("utilization.memory")),
        power_mean_w=statistics.mean(col("power.draw")),
        gpu_samples=len(gpu),
        artifacts=[r for r in rows if r["kind"] == "artifact"],
    )
    image_groups = {}
    for f in path.parent.glob("*.png"):
        if f.name.startswith(("cold-", "warm")):
            name = f.name.split("-", 1)[1]
            image_groups.setdefault(name, {})[f.name] = hashlib.sha256(
                f.read_bytes()
            ).hexdigest()
    reports[path.parent.name]["image_hashes"] = image_groups
(ROOT / "aggregate.json").write_text(json.dumps(reports, indent=2))
for name, r in reports.items():
    print(
        name,
        {k: r[k] for k in ("warm_mean_s", "nvml_peak_gib")},
        {k: round(v["e2e_s"], 3) for k, v in r["runs"].items()},
    )
