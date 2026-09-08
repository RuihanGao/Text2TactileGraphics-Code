"""Compare every generated artifact and Qwen call between two full workflows."""

import argparse
import json
from pathlib import Path

from compare_single_gpu import calls, image_comparison

p = argparse.ArgumentParser()
p.add_argument("reference", type=Path)
p.add_argument("candidate", type=Path)
p.add_argument("output", type=Path)
a = p.parse_args()
results = []
for x in sorted(a.reference.glob("cold-*")):
    y = a.candidate / x.name
    if not y.exists():
        results.append(dict(file=x.name, missing=True))
        continue
    if x.suffix == ".png":
        results.append(image_comparison(x, y))
    else:
        import hashlib

        results.append(
            dict(
                file=x.name,
                exact_file=x.read_bytes() == y.read_bytes(),
                reference_sha256=hashlib.sha256(x.read_bytes()).hexdigest(),
                candidate_sha256=hashlib.sha256(y.read_bytes()).hexdigest(),
            )
        )
comparisons = []
left, right = calls(a.reference), calls(a.candidate)
assert len(left) == len(right)
for i, (x, y) in enumerate(zip(left, right)):
    comparisons.append(
        dict(
            call=i,
            config_equal=x["before"]["config"] == y["before"]["config"],
            components_equal=x["before"]["components"] == y["before"]["components"],
            operations_equal=x["operations"] == y["operations"],
            scheduler_equal=x["after"]["scheduler"] == y["after"]["scheduler"],
        )
    )
out = dict(artifacts=results, qwen_calls=comparisons)
a.output.write_text(json.dumps(out, indent=2))
assert all(r.get("exact_file") for r in results), "Artifact mismatch"
assert all(all(v for k, v in r.items() if k != "call") for r in comparisons), (
    "Runtime mismatch"
)
print(len(results), "artifacts and", len(comparisons), "Qwen calls match exactly")
