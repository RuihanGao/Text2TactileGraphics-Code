"""Single-GPU validation using released handlers and the existing timing recorder."""

import argparse
import functools
import hashlib
import inspect
import json
import shutil
import subprocess
import traceback
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from profile_pipeline import Recorder, instrument, release, safe_text


def digest(data):
    return hashlib.sha256(data).hexdigest()


def describe(value):
    if isinstance(value, torch.Tensor):
        data = value.detach().contiguous().cpu()
        return dict(
            shape=list(data.shape),
            dtype=str(data.dtype),
            sha256=digest(data.view(torch.uint8).numpy().tobytes()),
        )
    if isinstance(value, Image.Image):
        return dict(
            size=list(value.size),
            mode=value.mode,
            pixels_sha256=digest(value.tobytes()),
        )
    if isinstance(value, dict):
        return {str(k): describe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe(v) for v in value]
    if value is None or isinstance(value, (str, float, int, bool)):
        return value
    return (
        str(value)
        if isinstance(value, (torch.dtype, torch.device, Path))
        else type(value).__name__
    )


def inventory(model):
    params = {}
    wrappers = {}
    adapters = {}
    for name, p in model.named_parameters():
        key = f"{p.dtype}/{p.device}"
        params[key] = params.get(key, 0) + p.numel()
    for name, m in model.named_modules():
        if hasattr(m, "computation_device"):
            fields = {
                k: str(getattr(m, k, None))
                for k in (
                    "offload_dtype",
                    "offload_device",
                    "onload_dtype",
                    "onload_device",
                    "preparing_dtype",
                    "preparing_device",
                    "computation_dtype",
                    "computation_device",
                )
            }
            key = json.dumps(fields, sort_keys=True)
            wrappers[key] = wrappers.get(key, 0) + 1
        if getattr(m, "lora_A_weights", []):
            key = str(
                [(str(a.dtype), str(a.device), list(a.shape)) for a in m.lora_A_weights]
            )
            adapters[key] = adapters.get(key, 0) + 1
    return dict(
        cls=type(model).__name__,
        parameters=params,
        wrappers=wrappers,
        adapters=adapters,
    )


def audit(rec):
    from diffsynth.diffusion.base_pipeline import BasePipeline
    from diffsynth.pipelines.qwen_image import QwenImagePipeline

    from text2tactilegraphics.generation.models import ModelManager

    stream = (rec.out / "runtime.jsonl").open("w", buffering=1)

    def emit(event, **data):
        stream.write(
            json.dumps(dict(run=rec.run, event=event, **describe(data))) + "\n"
        )

    old_load = ModelManager._load_qwen_pipeline

    @functools.wraps(old_load)
    def load(self, **kw):
        bundle = old_load(self, **kw)
        bundle["pipeline"]._audit_identity = kw
        emit(
            "load_identity",
            identity=kw,
            vram=self.config.get_qwen_vram_config(bundle["device"]),
        )
        return bundle

    ModelManager._load_qwen_pipeline = load
    old_pretrained = QwenImagePipeline.from_pretrained

    def pretrained(*a, **kw):
        result = old_pretrained(*a, **kw)
        emit(
            "resolved_checkpoints",
            configs=[vars(c) for c in kw["model_configs"]],
            tokenizer=vars(kw.get("tokenizer_config") or kw.get("processor_config")),
        )
        return result

    QwenImagePipeline.from_pretrained = staticmethod(pretrained)
    old_lora = BasePipeline.load_lora

    @functools.wraps(old_lora)
    def lora(self, *a, **kw):
        bound = inspect.signature(old_lora).bind(self, *a, **kw)
        bound.apply_defaults()
        result = old_lora(self, *a, **kw)
        emit(
            "lora_load",
            config={
                k: v
                for k, v in bound.arguments.items()
                if k not in ("self", "module", "state_dict")
            },
            hotload_effective=getattr(self.dit, "vram_management_enabled", False),
        )
        return result

    BasePipeline.load_lora = lora
    old_call = QwenImagePipeline.__call__

    @functools.wraps(old_call)
    def call(self, *a, **kw):
        bound = inspect.signature(old_call).bind(self, *a, **kw)
        bound.apply_defaults()
        emit(
            "qwen_before",
            identity=self._audit_identity,
            config={k: v for k, v in bound.arguments.items() if k != "self"},
            components={
                k: inventory(getattr(self, k)) for k in ("dit", "text_encoder", "vae")
            },
            scheduler_class=type(self.scheduler).__name__,
            scheduler_before=vars(self.scheduler),
        )
        with rec.span("qwen.inference", "inference"):
            result = old_call(self, *a, **kw)
        emit("qwen_after", image=result, scheduler=vars(self.scheduler))
        return result

    QwenImagePipeline.__call__ = call
    old_noise = BasePipeline.generate_noise

    @functools.wraps(old_noise)
    def noise(self, *a, **kw):
        result = old_noise(self, *a, **kw)
        emit("noise", arguments=kw, output=result)
        return result

    BasePipeline.generate_noise = noise
    old_step = BasePipeline.step

    @functools.wraps(old_step)
    def step(self, *a, **kw):
        result = old_step(self, *a, **kw)
        emit("denoising_step", progress_id=kw.get("progress_id"), output=result)
        return result

    BasePipeline.step = step
    # Capture deterministic auxiliary models at the actual infer/forward boundary.
    for label in ("moge2", "sam3_text"):
        desc = vars(ModelManager)[label]

        def wrap(fn, name):
            def wrapped(self):
                result = fn(self)
                emit("auxiliary_model", name=name, model=inventory(result["model"]))
                return result

            return wrapped

        prop = functools.cached_property(wrap(desc.func, label))
        prop.__set_name__(ModelManager, label)
        setattr(ModelManager, label, prop)
    return stream


def unload_qwen(state, mm):
    for model, owner in [
        ("qwen_base_edit", "_base_gen"),
        ("qwen_texture", "_texture_gen"),
        ("qwen_tiling", "_tiling_gens"),
    ]:
        state.__dict__.pop(owner, None)
        if model in mm.__dict__:
            mm.unload_model(model)


def workflow(args, state, mm, rec):
    from text2tactilegraphics.ui import handlers as h

    def save(image, name):
        image.save(rec.out / f"{rec.run}-{name}.png")

    with rec.span("handler.base_image"):
        image = h.generate_base_image(args.prompt, "qwen_edit", 4, 42, state)
    save(image, "base")
    unload_qwen(state, mm)
    segments = []
    for index, (segment_prompt, texture_prompt) in enumerate(args.regions):
        prefix = f"region{index + 1}"
        with rec.span("handler.segment"):
            mask, overlay = h.segment_with_text(image, segment_prompt, state)
        save(overlay, prefix + "-segmentation")
        np.save(rec.out / f"{rec.run}-{prefix}-mask.npy", mask)
        with rec.span("handler.texture"):
            texture = h.generate_texture_image(texture_prompt, 4, 42, state)
        save(texture, prefix + "-texture")
        if not args.reuse:
            unload_qwen(state, mm)
        with rec.span("handler.texture_geometry"):
            array, normal = h.generate_texture_geometry(texture, True, state)
        save(normal, prefix + "-normal")
        np.save(rec.out / f"{rec.run}-{prefix}-normal.npy", array)
        if args.reuse:
            # Hotloaded adapters are separate tensors; clearing never subtracts
            # rounded deltas from base weights. Refuse fused LoRA execution.
            pipe = mm.qwen_texture["pipeline"]
            assert pipe.dit.vram_management_enabled
            assert any(getattr(m, "lora_A_weights", []) for m in pipe.dit.modules())
            with rec.span("reuse.clear_adapters", "lora_setup"):
                state.get_texture_gen().loras.apply([])
                assert not any(
                    getattr(m, "lora_A_weights", []) for m in pipe.dit.modules()
                )
                mm.__dict__["qwen_tiling"] = mm.__dict__.pop("qwen_texture")
            del pipe
        with rec.span("handler.tiling"):
            patch = h.make_tileable(
                normal, "intra_tile_inpainting", 10, 42, True, 120, "per_channel", state
            )
        save(patch, prefix + "-patch")
        if args.reuse and index + 1 < len(args.regions):
            # Retain the clean base for the next saved region. The existing
            # LoraManager has empty loaded_paths and will reapply both adapters.
            with rec.span("reuse.next_region", "lifecycle"):
                mm.__dict__["qwen_texture"] = mm.__dict__.pop("qwen_tiling")
        else:
            unload_qwen(state, mm)
        with rec.span("handler.save_segment"):
            segments = h.save_segment(image, mask, patch, 0.005, "normal", 3, segments)
    with rec.span("handler.final_mesh"):
        path = h.generate_final_mesh(
            image,
            segments,
            "opengl",
            "standard",
            args.label,
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


def reference(args, state, rec):
    from text2tactilegraphics.ui import handlers as h

    if args.stage == "base":
        result = h.generate_base_image(args.prompt, "qwen_edit", 4, 42, state)
    elif args.stage == "texture":
        result = h.generate_texture_image(args.texture, 4, 42, state)
    else:
        result = h.make_tileable(
            Image.open(args.input).convert("RGB"),
            "intra_tile_inpainting",
            10,
            42,
            True,
            120,
            "per_channel",
            state,
        )
    result.save(rec.out / "reference.png")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--example", choices=["dolphin", "lamp"], default="dolphin")
    p.add_argument("--runs", type=int, default=1)
    p.add_argument("--stage", choices=["base", "texture", "tiling"])
    p.add_argument("--texture", default="an avocado skin")
    p.add_argument("--input", type=Path)
    p.add_argument("--audit", action="store_true")
    p.add_argument("--reuse", action="store_true")
    args = p.parse_args()
    args.prompt = "a dolphin with wings" if args.example == "dolphin" else "lamp"
    args.label = args.example
    args.regions = (
        [("dolphin", "an avocado skin")]
        if args.example == "dolphin"
        else [("lamp base", "tree bark"), ("lamp shade", "cloth_bag")]
    )
    args.output.mkdir(parents=True, exist_ok=False)
    shutil.copy(__file__, args.output / "harness.py")
    shutil.copy(
        Path(__file__).with_name("profile_pipeline.py"), args.output / "recorder.py"
    )
    assert torch.cuda.is_available() and torch.cuda.device_count() == 1
    from text2tactilegraphics.config import global_config
    from text2tactilegraphics.generation.models import global_model_manager
    from text2tactilegraphics.ui.state import AppState

    config = global_config()
    config.vram_mode = "80gb"
    mm = global_model_manager()
    state = AppState()
    (args.output / "settings.json").write_text(
        json.dumps(
            describe(
                dict(
                    args=vars(args),
                    config=vars(config),
                    git_sha=subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], text=True
                    ).strip(),
                    torch=torch.__version__,
                    cuda=torch.version.cuda,
                )
            ),
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
    try:
        for i in range(args.runs):
            rec.run = "cold" if i == 0 else f"warm{i}"
            with rec.span("end_to_end", "total"):
                if args.stage:
                    reference(args, state, rec)
                else:
                    workflow(args, state, mm, rec)
    except Exception:
        (args.output / "error.txt").write_text(safe_text(traceback.format_exc()))
        raise
    finally:
        rec.run = "cleanup"
        release(state, mm, rec)
        trace.terminate()
        trace.wait(timeout=10)
        trace_file.close()
        rec.stream.close()
        if stream:
            stream.close()


if __name__ == "__main__":
    main()
