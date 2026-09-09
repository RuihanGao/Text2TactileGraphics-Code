"""Real Quick Trial measurements and saved true-80GB pixel comparisons."""

import argparse
import functools
import hashlib
import json
import logging
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from PIL import Image

from text2tactilegraphics.generation.models import LoraManager, ModelManager
from text2tactilegraphics.ui import handlers
from text2tactilegraphics.ui.prompt_planner import GeminiPromptPlanner, PromptPlan
from text2tactilegraphics.ui.quick_pipeline import QuickPipeline, QuickState


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--warmup", action="store_true")
    parser.add_argument("--example", choices=["dolphin", "lamp"], default="dolphin")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO)
    (args.output / "source-sha256.json").write_text(
        json.dumps(
            {
                str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in Path("src/text2tactilegraphics").rglob("*.py")
            },
            indent=2,
        )
    )
    records = (args.output / "stages.jsonl").open("w", buffering=1)
    run = -1

    def memory():
        return [
            {
                k: getattr(torch.cuda, k)(i) / 1024**3
                for k in [
                    "memory_allocated",
                    "memory_reserved",
                    "max_memory_allocated",
                    "max_memory_reserved",
                ]
            }
            | {"free_gib": torch.cuda.mem_get_info(i)[0] / 1024**3}
            for i in range(torch.cuda.device_count())
        ]

    def sync():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)

    def wrap(obj, name):
        old = getattr(obj, name)

        @functools.wraps(old)
        def measured(*a, **kw):
            sync()
            start = time.perf_counter()
            try:
                return old(*a, **kw)
            finally:
                sync()
                records.write(
                    json.dumps(
                        {
                            "run": run,
                            "stage": name,
                            "seconds": time.perf_counter() - start,
                            "memory": memory(),
                        }
                    )
                    + "\n"
                )

        setattr(obj, name, measured)

    for name in [
        "generate_base_image",
        "segment_with_text",
        "generate_texture_image",
        "generate_texture_geometry",
        "make_tileable",
        "generate_final_mesh",
    ]:
        wrap(handlers, name)
    wrap(ModelManager, "_load_qwen_pipeline")
    wrap(LoraManager, "apply")
    wrap(GeminiPromptPlanner, "plan")
    is_lamp = args.example == "lamp"
    prompt = (
        "lamp, the base of the lamp has a tree bark texture, and the lamp shade has a cloth_bag texture."
        if is_lamp
        else "a dolphin with wings with an avocado skin texture."
    )

    class ReferencePlanner:
        def plan(self, _):
            return PromptPlan(
                shape_prompt="lamp" if is_lamp else "a dolphin with wings",
                braille_label=args.example,
                regions=[
                    {"part_prompt": p, "texture_prompt": t}
                    for p, t in (
                        [("lamp base", "tree bark"), ("lamp shade", "cloth_bag")]
                        if is_lamp
                        else [("dolphin", "an avocado skin")]
                    )
                ],
            )

    pipe = QuickPipeline(GeminiPromptPlanner() if args.live else ReferencePlanner())
    trace_file = (args.output / "gpu.csv").open("w")
    trace = subprocess.Popen(
        [
            "nvidia-smi",
            "--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.free,power.draw",
            "--format=csv,nounits",
            "-l",
            "1",
        ],
        stdout=trace_file,
    )
    reference = Path(
        "/workspace/Text2TactileGraphics-Code/profiling/results/single-gpu-quality"
    ) / ("lamp-reuse-validation" if is_lamp else "dolphin-performance")
    results = []
    try:
        if args.warmup:
            from text2tactilegraphics.generation.models import global_model_manager

            start = time.perf_counter()
            global_model_manager().warmup()
            (args.output / "warmup.json").write_text(
                json.dumps(
                    {"seconds": time.perf_counter() - start, "memory": memory()},
                    indent=2,
                )
            )
        for run in range(args.runs):
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
            s = QuickState()
            sync()
            start = time.perf_counter()
            events = []
            for state in pipe.generate(s, prompt):
                events.append(
                    {"seconds": time.perf_counter() - start, "status": state.status}
                )
            sync()
            elapsed = time.perf_counter() - start
            dest = args.output / str(run)
            dest.mkdir(exist_ok=True)
            artifacts = {"base": s.base_image}
            for index, region in enumerate(s.regions, 1):
                artifacts.update(
                    {
                        f"region{index}-texture": region.texture,
                        f"region{index}-patch": region.tiled,
                        f"region{index}-normal": region.geometry,
                    }
                )
                np.save(dest / f"region{index}-mask.npy", region.mask)
            comparisons = {}
            for name, image in artifacts.items():
                image.save(dest / f"{name}.png")
                if not args.live:
                    expected = np.asarray(
                        Image.open(reference / f"cold-{name}.png")
                    ).astype(float)
                    diff = np.abs(np.asarray(image).astype(float) - expected)
                    comparisons[name] = {
                        "exact": bool(not diff.any()),
                        "max_abs": float(diff.max()),
                        "mean_abs": float(diff.mean()),
                    }
            if not args.live:
                for index, region in enumerate(s.regions, 1):
                    expected = np.load(reference / f"cold-region{index}-mask.npy")
                    comparisons[f"region{index}-mask"] = {
                        "exact": bool(np.array_equal(expected, region.mask))
                    }
            shutil.copyfile(s.final_mesh, dest / "final.glb")
            row = {
                "run": run,
                "rss_gib": psutil.Process().memory_info().rss / 1024**3,
                "seconds": elapsed,
                "memory": memory(),
                "comparisons": comparisons,
                "plan": s.plan.model_dump(),
                "events": events,
            }
            results.append(row)
            (args.output / "results.json").write_text(json.dumps(results, indent=2))
            print(
                json.dumps(
                    {"run": run, "seconds": elapsed, "comparisons": comparisons}
                ),
                flush=True,
            )
            if comparisons and not all(x["exact"] for x in comparisons.values()):
                raise RuntimeError(
                    "Numerical divergence: see saved differences; stop optimization acceptance"
                )
    finally:
        trace.terminate()
        trace.wait()
        trace_file.close()
        records.close()


if __name__ == "__main__":
    main()
