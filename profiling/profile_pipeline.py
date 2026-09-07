"""Instrument the released handlers; never replace generation algorithms."""

import argparse
import contextlib
import functools
import hashlib
import json
import os
import resource
import shutil
import subprocess
import time
import traceback
from pathlib import Path

import torch

GIB = 1024**3


def memory():
    free, total = torch.cuda.mem_get_info()
    return {
        k: v / GIB
        for k, v in {
            "allocated": torch.cuda.memory_allocated(),
            "reserved": torch.cuda.memory_reserved(),
            "peak_allocated": torch.cuda.max_memory_allocated(),
            "peak_reserved": torch.cuda.max_memory_reserved(),
            "free": free,
            "total": total,
        }.items()
    }


def safe_text(value):
    text = str(value)
    for key, secret in os.environ.items():
        if (
            any(
                word in key.upper()
                for word in ("TOKEN", "SECRET", "API_KEY", "PASSWORD")
            )
            and len(secret) > 5
        ):
            text = text.replace(secret, "[REDACTED]")
    return text


class Recorder:
    def __init__(self, out):
        self.out = out
        self.stream = (out / "stages.jsonl").open("w", buffering=1)
        self.stack = []
        self.run = "setup"
        self.counter = 0
        self.mm = None

    def emit(self, row):
        row["process_peak_rss_gib"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024**2
        )
        self.stream.write(json.dumps(row) + "\n")

    @contextlib.contextmanager
    def span(self, name, kind="stage"):
        torch.cuda.synchronize()
        # Preserve peaks of a parent segment before a child resets CUDA counters.
        if self.stack:
            prior = memory()
            for key in ("peak_allocated", "peak_reserved"):
                self.stack[-1]["peaks"][key] = max(
                    self.stack[-1]["peaks"][key], prior[key]
                )
        before = memory()
        torch.cuda.reset_peak_memory_stats()
        self.counter += 1
        row = dict(
            id=self.counter,
            parent=self.stack[-1]["id"] if self.stack else None,
            run=self.run,
            name=name,
            kind=kind,
            start_unix=time.time(),
            before_gib=before,
        )
        frame = dict(
            id=row["id"],
            children_s=0.0,
            peaks={
                "peak_allocated": before["allocated"],
                "peak_reserved": before["reserved"],
            },
        )
        self.stack.append(frame)
        start = time.perf_counter()
        try:
            yield
            row["status"] = "ok"
        except BaseException as exc:
            row.update(
                status="error", error=safe_text(exc), exception=type(exc).__name__
            )
            raise
        finally:
            try:
                torch.cuda.synchronize()
            except RuntimeError:
                pass
            elapsed = time.perf_counter() - start
            after = memory()
            for key in frame["peaks"]:
                after[key] = max(after[key], frame["peaks"][key])
            self.stack.pop()
            if self.stack:
                self.stack[-1]["children_s"] += elapsed
                for key in frame["peaks"]:
                    self.stack[-1]["peaks"][key] = max(
                        self.stack[-1]["peaks"][key], after[key]
                    )
            row.update(
                seconds=elapsed,
                exclusive_seconds=elapsed - frame["children_s"],
                after_gib=after,
                resident_models=sorted(self.mm._model_names() & self.mm.__dict__.keys())
                if self.mm
                else [],
            )
            self.emit(row)

    def wrap(self, owner, attr, name=None, kind="stage"):
        original = getattr(owner, attr)

        @functools.wraps(original)
        def wrapped(*a, **kw):
            with self.span(name or attr, kind):
                return original(*a, **kw)

        setattr(owner, attr, wrapped)


def instrument(rec):
    import trimesh
    from diffsynth.diffusion.base_pipeline import BasePipeline

    from text2tactilegraphics.generation.base_image_generation import BaseImageGenerator
    from text2tactilegraphics.generation.models import LoraManager, ModelManager
    from text2tactilegraphics.generation.segmentation import SegmentationEngine
    from text2tactilegraphics.generation.texture_generation import (
        GeometryEstimator,
        TextureGenerator,
    )
    from text2tactilegraphics.generation.tileable_patch_generation import (
        IntraTilePatchGenerator,
        TiledDiffusion,
    )
    from text2tactilegraphics.geometry import tactile_graphics as tg

    rec.wrap(
        BasePipeline,
        "load_models_to_device",
        "qwen.device_transition",
        "device_transition",
    )
    # Preserve cached_property behavior and wrap only the actual cache-miss load.
    for name in ModelManager._model_names():
        descriptor = vars(ModelManager)[name]

        def build(fn, label):
            @functools.wraps(fn)
            def wrapped(self):
                with rec.span("load." + label, "model_load"):
                    return fn(self)

            return wrapped

        replacement = functools.cached_property(build(descriptor.func, name))
        replacement.__set_name__(ModelManager, name)
        setattr(ModelManager, name, replacement)
    for owner, method, label, kind in [
        (LoraManager, "apply", "lora.apply", "lora_setup"),
        (BaseImageGenerator, "generate", "base_image", "stage"),
        (TextureGenerator, "generate", "texture_image", "stage"),
        (GeometryEstimator, "compute_depth", "depth", "stage"),
        (GeometryEstimator, "compute_normal", "normal", "stage"),
        (SegmentationEngine, "segment_with_text", "segmentation", "stage"),
        (IntraTilePatchGenerator, "make_tileable", "tileable_texture", "stage"),
        (TiledDiffusion, "make_tileable", "sdxl_tileable_texture", "stage"),
        (tg, "depth2mesh", "mesh_construction", "stage"),
        (tg, "pv2trimesh", "mesh_conversion", "stage"),
        (tg, "apply_segment_displacements", "texture_integration", "stage"),
        (tg, "flatten_and_close_plate", "plate_closure", "stage"),
        (tg, "_apply_braille", "braille", "stage"),
        (trimesh.Trimesh, "export", "export", "stage"),
        (ModelManager, "unload_model", "unload_model", "unload"),
        (ModelManager, "unload_all_models", "unload_all_models", "unload"),
    ]:
        rec.wrap(owner, method, label, kind)


def release(state, mm, rec):
    # Generator LoraManagers also retain pipelines. Release these owners before
    # exercising the existing ModelManager unload API; do not change its code.
    for attr in (
        "_base_gen",
        "_texture_gen",
        "_geom_estimator",
        "_seg_engine",
        "_tiling_gens",
    ):
        state.__dict__.pop(attr, None)
    mm.unload_all_models()


def pipeline(args, state, mm, rec):
    import trimesh

    from text2tactilegraphics.ui import handlers as h

    def unload():
        if args.lifecycle == "unload":
            release(state, mm, rec)
        elif args.lifecycle == "unload-qwen":
            for model, owner in (
                ("qwen_base_edit", "_base_gen"),
                ("qwen_texture", "_texture_gen"),
                ("qwen_tiling", "_tiling_gens"),
            ):
                if model in mm.__dict__:
                    state.__dict__.pop(owner, None)
                    mm.unload_model(model)

    with rec.span("handler.base_image"):
        image = h.generate_base_image(
            args.prompt, "qwen_edit", args.steps, args.seed, state
        )
    image.save(rec.out / f"{rec.run}-base.png")
    unload()
    if args.base_preview:
        with rec.span("handler.base_preview"):
            preview = h.generate_base_mesh(image)
        shutil.move(preview, rec.out / f"{rec.run}-base.glb")
    with rec.span("handler.segment"):
        mask, _ = h.segment_with_text(image, "dolphin", state)
    unload()
    with rec.span("handler.texture"):
        texture = h.generate_texture_image(
            "an avocado skin", args.steps, args.seed, state
        )
    texture.save(rec.out / f"{rec.run}-texture.png")
    unload()
    with rec.span("handler.texture_geometry"):
        _, normals = h.generate_texture_geometry(texture, True, state)
    unload()
    with rec.span("handler.tiling"):
        patch = h.make_tileable(
            normals,
            "intra_tile_inpainting",
            10,
            args.seed,
            True,
            120,
            "per_channel",
            state,
        )
    patch.save(rec.out / f"{rec.run}-patch.png")
    unload()
    with rec.span("handler.save_segment"):
        segments = h.save_segment(image, mask, patch, 0.005, "normal", 3, [])
    with rec.span("handler.final_mesh"):
        path = h.generate_final_mesh(
            image,
            segments,
            "opengl",
            "standard",
            "dolphin",
            0.12,
            0.3,
            0.001,
            [],
            0.0005,
            True,
            0.004,
        )
    dest = rec.out / f"{rec.run}-final.glb"
    shutil.move(path, dest)
    with rec.span("validate_export", "validation"):
        mesh = trimesh.load_mesh(dest)
        info = dict(
            watertight=bool(mesh.is_watertight),
            vertices=len(mesh.vertices),
            faces=len(mesh.faces),
            file=dest.name,
        )
        rec.emit(dict(run=rec.run, kind="artifact", **info))
    return info


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["80gb", "48gb"], required=True)
    p.add_argument("--steps", type=int, choices=[4, 40], default=4)
    p.add_argument("--runs", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--prompt", default="a dolphin with wings")
    p.add_argument(
        "--lifecycle", choices=["natural", "unload", "unload-qwen"], default="natural"
    )
    p.add_argument(
        "--base-preview",
        action="store_true",
        help="Include optional UI base-mesh preview",
    )
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    source = Path(__file__).read_bytes()
    (args.output / "harness.py").write_bytes(source)
    (args.output / "harness.sha256").write_text(
        hashlib.sha256(source).hexdigest() + "\n"
    )
    (args.output / "settings.json").write_text(
        json.dumps({**vars(args), "output": str(args.output)}, indent=2) + "\n"
    )
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Benchmark requires exactly one visible CUDA GPU")
    from text2tactilegraphics.config import global_config
    from text2tactilegraphics.generation.models import global_model_manager
    from text2tactilegraphics.ui.state import AppState

    config = global_config()
    config.vram_mode = args.mode
    mm = global_model_manager()
    state = AppState()
    rec = Recorder(args.output)
    rec.mm = mm
    instrument(rec)
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
        stderr=subprocess.DEVNULL,
    )
    success = True
    try:
        for i in range(args.runs):
            rec.run = "cold" if i == 0 else f"warm{i}"
            try:
                with rec.span("end_to_end", "total"):
                    pipeline(args, state, mm, rec)
            except Exception:
                (args.output / f"{rec.run}-error.txt").write_text(
                    safe_text(traceback.format_exc())
                )
                success = False
                # A failed cold run is not a warm baseline. Preserve partial
                # stages; retry only in a fresh explicitly labeled experiment.
                break
    finally:
        rec.run = "cleanup"
        release(state, mm, rec)
        trace.terminate()
        trace.wait(timeout=10)
        trace_file.close()
        rec.stream.close()
    return 0 if success else 1


if __name__ == "__main__":
    raise SystemExit(main())
