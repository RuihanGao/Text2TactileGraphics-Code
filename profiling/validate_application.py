"""Validate real Gradio callbacks; lifecycle belongs entirely to application code.

Only timing and optional read-only numerical audit hooks come from profiling.
No model eviction, owner deletion, cache aliasing, or config patching occurs here.
"""

import argparse
import hashlib
import json
import logging
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch
from profile_pipeline import Recorder, instrument
from validate_single_gpu import audit, describe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--example", choices=["dolphin", "lamp"], default="dolphin")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--audit", action="store_true")
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO)

    from text2tactilegraphics.generation.models import global_model_manager
    from text2tactilegraphics.ui import app, handlers
    from text2tactilegraphics.ui.state import AppState

    demo = app.create_demo()
    callbacks = {f.api_name: f for f in demo.fns.values()}
    base_callback = callbacks["generate_base_image"].fn
    state = next(
        c.cell_contents
        for c in base_callback.__closure__
        if isinstance(c.cell_contents, AppState)
    )
    mm = global_model_manager()
    assert state.config.model_lifecycle == "single_a100_80gb"
    assert state.config.vram_mode == "80gb"
    assert torch.cuda.device_count() == 1
    queued = [f for f in demo.fns.values() if f.concurrency_id == "single_a100_80gb"]
    assert len(queued) == 9 and all(f.concurrency_limit == 1 for f in queued)
    (args.output / "settings.json").write_text(
        json.dumps(
            describe(
                dict(
                    args=vars(args),
                    config=vars(state.config),
                    git_sha=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    branch=subprocess.check_output(
                        ["git", "branch", "--show-current"], text=True
                    ).strip(),
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                    gpu=torch.cuda.get_device_name(),
                    total_vram_gib=torch.cuda.mem_get_info()[1] / 1024**3,
                    queued_callbacks=[f.api_name for f in queued],
                )
            ),
            indent=2,
        )
    )
    shutil.copy(__file__, args.output / "validator.py")
    source_files = list(Path("src/text2tactilegraphics").rglob("*.py"))
    (args.output / "source-sha256.json").write_text(
        json.dumps(
            {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in source_files},
            indent=2,
        )
    )
    rec = Recorder(args.output)
    rec.mm = mm
    instrument(rec)
    stream = audit(rec) if args.audit else None
    if not args.audit:
        from diffsynth.pipelines.qwen_image import QwenImagePipeline

        rec.wrap(QwenImagePipeline, "__call__", "qwen.inference", "inference")
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
    prompt = "a dolphin with wings" if args.example == "dolphin" else "lamp"
    regions = (
        [("dolphin", "an avocado skin")]
        if args.example == "dolphin"
        else [("lamp base", "tree bark"), ("lamp shade", "cloth_bag")]
    )

    def call(name, *values):
        with rec.span("callback." + name):
            return callbacks[name].fn(*values)

    def save(image, name):
        image.save(args.output / f"{rec.run}-{name}.png")

    def assert_released():
        assert not any(k.startswith("qwen_") for k in mm.__dict__)
        for key in ("_base_gen", "_texture_gen"):
            owner = state.__dict__.get(key)
            assert owner is None or "loras" not in owner.__dict__

    try:
        for i in range(args.runs):
            rec.run = "cold" if i == 0 else f"warm{i}"
            with rec.span("end_to_end", "total"):
                image = call("generate_base_image", prompt, "qwen_edit", 4, 42)
                assert_released()
                save(image, "base")
                segments = []
                for index, (segment, texture_prompt) in enumerate(regions, 1):
                    prefix = f"region{index}"
                    with rec.span("handler.segment"):
                        mask, overlay = handlers.segment_with_text(
                            image, segment, state
                        )
                    save(overlay, prefix + "-segmentation")
                    np.save(args.output / f"{rec.run}-{prefix}-mask.npy", mask)
                    texture = call("generate_texture_image", texture_prompt, 4, 42)
                    assert_released()
                    save(texture, prefix + "-texture")
                    array, normal = call("generate_texture_geometry", texture, True)
                    save(normal, prefix + "-normal")
                    np.save(args.output / f"{rec.run}-{prefix}-normal.npy", array)
                    patch = call(
                        "make_tileable",
                        normal,
                        "intra_tile_inpainting",
                        True,
                        120,
                        "per_channel",
                    )
                    assert_released()
                    save(patch, prefix + "-patch")
                    with rec.span("handler.save_segment"):
                        segments = handlers.save_segment(
                            image, mask, patch, 0.005, "normal", 3, segments
                        )
                path = call(
                    "generate_final_mesh",
                    image,
                    segments,
                    "opengl",
                    "standard",
                    args.example,
                    12,
                    0.3,
                    0.1,
                    [],
                    0.5,
                    True,
                    4,
                )
                dest = args.output / f"{rec.run}-final.glb"
                shutil.move(path, dest)
                import trimesh

                with rec.span("validate_export", "validation"):
                    mesh = trimesh.load_mesh(dest)
                    rec.emit(
                        dict(
                            run=rec.run,
                            kind="artifact",
                            segments=len(segments),
                            watertight=bool(mesh.is_watertight),
                            vertices=len(mesh.vertices),
                            faces=len(mesh.faces),
                        )
                    )
                assert_released()
    finally:
        trace.terminate()
        trace.wait(timeout=10)
        trace_file.close()
        rec.stream.close()
        if stream:
            stream.close()
        demo.close()


if __name__ == "__main__":
    main()
