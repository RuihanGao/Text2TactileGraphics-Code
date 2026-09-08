"""Compare fresh true-80GB references with lifecycle outputs and runtime evidence."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def image_comparison(a, b):
    x, y = (
        np.asarray(Image.open(a)).astype(np.int16),
        np.asarray(Image.open(b)).astype(np.int16),
    )
    result = dict(
        reference=str(a),
        candidate=str(b),
        reference_sha256=hashlib.sha256(a.read_bytes()).hexdigest(),
        candidate_sha256=hashlib.sha256(b.read_bytes()).hexdigest(),
        shape_equal=x.shape == y.shape,
    )
    if x.shape == y.shape:
        diff = np.abs(x - y)
        result.update(
            exact_pixels=bool(np.array_equal(x, y)),
            max_absolute_pixel_difference=int(diff.max()),
            mean_absolute_pixel_difference=float(diff.mean()),
        )
    result["exact_file"] = result["reference_sha256"] == result["candidate_sha256"]
    return result


def calls(path):
    result = []
    active = None
    for line in (path / "runtime.jsonl").read_text().splitlines():
        row = json.loads(line)
        if row["event"] == "qwen_before":
            active = dict(before=row, operations=[])
        elif row["event"] in ("noise", "denoising_step") and active is not None:
            active["operations"].append(
                {k: v for k, v in row.items() if k not in ("run",)}
            )
        elif row["event"] == "qwen_after":
            active["after"] = row
            result.append(active)
            active = None
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--references", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    candidate = calls(args.candidate)
    rows = []
    for i, (name, suffix) in enumerate(
        [("base", "base"), ("texture", "region1-texture"), ("tiling", "region1-patch")]
    ):
        reference = args.references / f"reference-{name}"
        row = image_comparison(
            reference / "reference.png", args.candidate / f"cold-{suffix}.png"
        )
        a, b = calls(reference)[0], candidate[i]
        row.update(
            stage=name,
            inference_config_equal=a["before"]["config"] == b["before"]["config"],
            component_dtype_device_adapters_equal=a["before"]["components"]
            == b["before"]["components"],
            scheduler_after_equal=a["after"]["scheduler"] == b["after"]["scheduler"],
            operations_equal=a["operations"] == b["operations"],
        )
        first = None
        for j, (x, y) in enumerate(zip(a["operations"], b["operations"])):
            if x != y:
                first = dict(index=j, reference=x, candidate=y)
                break
        if len(a["operations"]) != len(b["operations"]):
            first = first or dict(reason="operation count differs")
        row["first_recorded_divergence"] = first
        rows.append(row)
    args.output.write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(
            row["stage"],
            {
                k: row[k]
                for k in (
                    "exact_file",
                    "inference_config_equal",
                    "component_dtype_device_adapters_equal",
                    "scheduler_after_equal",
                    "operations_equal",
                )
            },
        )
    return (
        0
        if all(
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
        else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
