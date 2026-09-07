"""Aggregate real JSONL spans without adding overlapping inclusive timings."""

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).parent / "results"


def mean(values):
    return statistics.mean(values) if values else None


def main():
    reports = {}
    for path in sorted(ROOT.glob("*/stages.jsonl")):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        timed = [r for r in rows if "seconds" in r]
        totals = [r for r in timed if r["kind"] == "total"]
        if not totals:
            continue
        finished_runs = {r["run"] for r in totals}
        timed = [r for r in timed if r["run"] in finished_runs or r["run"] == "cleanup"]
        stages = {}
        for name in dict.fromkeys(r["name"] for r in timed):
            group = [r for r in timed if r["name"] == name]
            by_run = {}
            for r in group:
                if r["run"] == "cleanup":
                    continue
                record = by_run.setdefault(
                    r["run"],
                    {
                        "inclusive_s": 0,
                        "exclusive_s": 0,
                        "count": 0,
                        "status": "ok",
                        "peak_allocated_gib": 0,
                        "resident_allocated_gib": 0,
                    },
                )
                record["inclusive_s"] += r["seconds"]
                record["exclusive_s"] += r["exclusive_seconds"]
                record["count"] += 1
                if r["status"] != "ok":
                    record["status"] = r["status"]
                record["peak_allocated_gib"] = max(
                    record["peak_allocated_gib"], r["after_gib"]["peak_allocated"]
                )
                record["resident_allocated_gib"] = r["after_gib"]["allocated"]
            stages[name] = by_run
        gpu_path = path.parent / "gpu.csv"
        gpu = list(csv.DictReader(gpu_path.open())) if gpu_path.exists() else []

        def column(suffix):
            if not gpu:
                return []
            key = next(k for k in gpu[0] if k.strip().startswith(suffix))
            return [
                float(r[key])
                for r in gpu
                if r.get(key, "").strip() not in ("", "[N/A]")
            ]

        warm = [
            r["seconds"]
            for r in totals
            if r["run"].startswith("warm") and r["status"] == "ok"
        ]
        reports[path.parent.name] = dict(
            process_peak_rss_gib=max(
                (r.get("process_peak_rss_gib", 0) for r in timed), default=0
            ),
            settings=json.loads((path.parent / "settings.json").read_text()),
            completed_runs=[r["run"] for r in totals if r["status"] == "ok"],
            failed_runs=[
                {"run": r["run"], "error": r.get("error")}
                for r in totals
                if r["status"] != "ok"
            ],
            warm_mean_s=mean(warm),
            warm_min_s=min(warm) if warm else None,
            warm_max_s=max(warm) if warm else None,
            cold_s=next((r["seconds"] for r in totals if r["run"] == "cold"), None),
            peak_allocated_gib=max(
                (r["after_gib"]["peak_allocated"] for r in timed), default=0
            ),
            peak_reserved_gib=max(
                (r["after_gib"]["peak_reserved"] for r in timed), default=0
            ),
            gpu_trace=dict(
                samples=len(gpu),
                mean_gpu_util_pct=mean(column("utilization.gpu")),
                mean_memory_util_pct=mean(column("utilization.memory")),
                max_used_gib=max(column("memory.used"), default=0) / 1024,
                mean_power_w=mean(column("power.draw")),
            ),
            stages=stages,
            artifacts=[r for r in rows if r["kind"] == "artifact"],
        )
        # All direct children fit within their parent's synchronized interval.
        for r in timed:
            assert r["exclusive_seconds"] >= -0.001, r
            if r["parent"] is not None:
                parent = next(x for x in timed if x["id"] == r["parent"])
                assert (
                    r["after_gib"]["peak_allocated"]
                    <= parent["after_gib"]["peak_allocated"] + 1e-6
                )
    host_path = ROOT / "host-trace.jsonl"
    if host_path.exists():
        host_rows = [json.loads(line) for line in host_path.read_text().splitlines()]
        for name, report in reports.items():
            spans = [
                json.loads(line)
                for line in (ROOT / name / "stages.jsonl").read_text().splitlines()
            ]
            spans = [s for s in spans if s.get("kind") == "total"]
            start = min(s["start_unix"] for s in spans)
            end = max(s["start_unix"] + s["seconds"] for s in spans)
            samples = [h for h in host_rows if start <= h["unix"] <= end]
            if len(samples) >= 2:

                def cpu(h):
                    return dict(
                        (k, int(v))
                        for k, v in (
                            line.split() for line in h["cpu/cpu.stat"].splitlines()
                        )
                    )

                first, last = cpu(samples[0]), cpu(samples[-1])
                periods = last["nr_periods"] - first["nr_periods"]
                report["host_trace"] = {
                    "sampled_seconds": samples[-1]["unix"] - samples[0]["unix"],
                    "experiment_seconds": end - start,
                    "cpu_throttled_period_fraction": (
                        last["nr_throttled"] - first["nr_throttled"]
                    )
                    / periods
                    if periods
                    else None,
                    "peak_cgroup_memory_gib": max(
                        int(h["memory/memory.usage_in_bytes"]) for h in samples
                    )
                    / 1024**3,
                }
    (ROOT / "aggregate.json").write_text(json.dumps(reports, indent=2) + "\n")
    lines = [
        "# Measured stages",
        "",
        "Inclusive spans overlap; exclusive timings subtract immediate child spans. Times in seconds.",
        "",
    ]
    for name, report in reports.items():
        lines += [
            f"## {name}",
            "",
            "| Stage | Cold inclusive | Warm mean inclusive | Cold exclusive | Warm mean exclusive |",
            "|---|---:|---:|---:|---:|",
        ]
        for stage, runs in report["stages"].items():
            if not runs:
                continue
            cold = runs.get("cold", {})
            warm = [
                r
                for k, r in runs.items()
                if k.startswith("warm") and r["status"] == "ok"
            ]

            def fmt(x):
                return "—" if x is None else f"{x:.3f}"

            lines.append(
                f"| {stage} | {fmt(cold.get('inclusive_s'))} | {fmt(mean([r['inclusive_s'] for r in warm]))} | {fmt(cold.get('exclusive_s'))} | {fmt(mean([r['exclusive_s'] for r in warm]))} |"
            )
        lines.append("")
    (ROOT / "STAGE_TABLES.md").write_text("\n".join(lines))
    for name, r in reports.items():
        print(
            name,
            "completed",
            r["completed_runs"],
            "cold",
            r["cold_s"],
            "warm",
            r["warm_mean_s"],
            "peak",
            r["peak_allocated_gib"],
        )


if __name__ == "__main__":
    main()
