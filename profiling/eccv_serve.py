"""Observe the real application; no lifecycle or handler substitutions."""

import functools
import gc
import hashlib
import json
import os
import runpy
import threading
import time
from pathlib import Path

import psutil
import torch

from text2tactilegraphics.generation.models import LoraManager, ModelManager
from text2tactilegraphics.geometry import tactile_graphics as tg
from text2tactilegraphics.ui import handlers
from text2tactilegraphics.ui.prompt_planner import GeminiPromptPlanner, PlannerError

out = Path(os.environ["ECCV_MEASUREMENT_DIR"])
out.mkdir(parents=True, exist_ok=True)
(out / "source-sha256.json").write_text(
    json.dumps(
        {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in Path("src/text2tactilegraphics").rglob("*.py")
        },
        indent=2,
    )
)
stream = (out / "stages.jsonl").open("a", buffering=1)
local = threading.local()


def sample_peaks():
    return [
        [torch.cuda.max_memory_allocated(i), torch.cuda.max_memory_reserved(i)]
        for i in range(torch.cuda.device_count())
    ]


def merge_peaks(target, source):
    for dst, src in zip(target, source):
        dst[:] = [max(a, b) for a, b in zip(dst, src)]


def wrap(obj, name):
    old = getattr(obj, name)

    @functools.wraps(old)
    def measured(*a, **kw):
        gpu = obj is not GeminiPromptPlanner
        stack = getattr(local, "stack", [])
        local.stack = stack
        peaks = [[0, 0] for _ in range(torch.cuda.device_count())]
        if gpu:
            for i in range(torch.cuda.device_count()):
                torch.cuda.synchronize(i)
            if stack:
                merge_peaks(stack[-1], sample_peaks())
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
            stack.append(peaks)
        start = time.perf_counter()
        rss_before = psutil.Process().memory_info().rss / 1024**3
        error = None
        injected = False
        result = None
        try:
            fault = out / f"fault-{name}"
            if os.getenv("ECCV_TEST_FAULTS") == "1" and fault.exists():
                fault.rename(fault.with_suffix(".consumed"))
                injected = True
                if name == "plan":
                    raise PlannerError(
                        "The text planner is temporarily unavailable. Please try again."
                    )
                raise RuntimeError("Injected generation failure")
            result = old(*a, **kw)
            return result
        except Exception as exc:
            error = type(exc).__name__
            raise
        finally:
            if gpu:
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)
                merge_peaks(peaks, sample_peaks())
                stack.pop()
                if stack:
                    merge_peaks(stack[-1], peaks)
            stream.write(
                json.dumps(
                    {
                        "timestamp": time.time(),
                        "stage": name,
                        "seconds": time.perf_counter() - start,
                        "error_class": error,
                        "test_injected": injected,
                        "rss_gib": psutil.Process().memory_info().rss / 1024**3,
                        "rss_before_gib": rss_before,
                        "collected_objects": result if obj is gc else None,
                        "gpu": [
                            {
                                "allocated_gib": torch.cuda.memory_allocated(i)
                                / 1024**3,
                                "reserved_gib": torch.cuda.memory_reserved(i) / 1024**3,
                                "peak_allocated_gib": peaks[i][0] / 1024**3,
                                "peak_reserved_gib": peaks[i][1] / 1024**3,
                                "free_gib": torch.cuda.mem_get_info(i)[0] / 1024**3,
                            }
                            for i in range(torch.cuda.device_count())
                        ],
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
wrap(gc, "collect")
for name in [
    "depth2mesh",
    "pv2trimesh",
    "apply_segment_displacements",
    "flatten_and_close_plate",
    "_apply_braille",
]:
    wrap(tg, name)
runpy.run_module("text2tactilegraphics.ui.public_app", run_name="__main__")
