"""Regression checks for release, retries, and serial Qwen execution."""

import threading
import time
import weakref
from concurrent.futures import ThreadPoolExecutor

import pytest

from text2tactilegraphics.config import GPU_ASSIGNMENTS, Config
from text2tactilegraphics.generation.models import LoraManager, ModelManager, qwen_stage


class Pipeline:
    def __init__(self):
        self.dit = Component()
        self.text_encoder = Component()
        self.vae = Component()


class Component:
    pass


def config(lifecycle="single_a100_80gb"):
    return Config(
        model_lifecycle=lifecycle,
        vram_mode="80gb",
        gpu_assignments=dict.fromkeys(GPU_ASSIGNMENTS, 0),
    )


class Generator:
    def __init__(self, mm):
        self.mm = mm
        self.config = mm.config
        self.references = []

    @qwen_stage("qwen_texture")
    def generate(self, fail=False):
        pipe = Pipeline()
        self.references.append(weakref.ref(pipe))
        self.mm.__dict__["qwen_texture"] = {"pipeline": pipe, "device": "cuda:0"}
        self.loras = LoraManager(pipe, self.config)
        if fail:
            raise ValueError("inference failed")
        return "image"


def test_release_and_retry_preserve_auxiliary_models():
    mm = ModelManager(config())
    aux = {"model": Component()}
    mm.__dict__["moge2"] = aux
    mm.__dict__["sam3_text"] = aux
    gen = Generator(mm)
    for fail in (False, True, False):
        if fail:
            with pytest.raises(ValueError, match="inference failed") as error:
                gen.generate(fail=True)
            # Keep the exception alive: its traceback must not retain weights.
            assert error.value.__traceback__ is not None
        else:
            assert gen.generate() == "image"
        assert gen.references[-1]() is None
        assert "loras" not in gen.__dict__
        assert "qwen_texture" not in mm.__dict__
        assert mm.moge2 is aux and mm.sam3_text is aux


def test_cached_policy_keeps_existing_behavior():
    mm = ModelManager(config("cached"))
    gen = Generator(mm)
    gen.generate()
    assert gen.references[-1]() is gen.loras.pipeline
    assert "qwen_texture" in mm.__dict__


def test_rejects_48gb_and_other_gpu_assignments():
    with pytest.raises(ValueError, match="true 80gb"):
        Config(model_lifecycle="single_a100_80gb", vram_mode="48gb")
    with pytest.raises(ValueError, match="cuda:0"):
        Config(
            model_lifecycle="single_a100_80gb",
            vram_mode="80gb",
            gpu_assignments={"qwen_texture": 1},
        )
    cfg = config()
    cfg.vram_mode = "48gb"
    with pytest.raises(ValueError, match="true 80gb"):
        cfg.get_qwen_vram_config("cuda:0")


def test_external_owner_is_reported():
    mm = ModelManager(config())
    gen = Generator(mm)
    pipe = Pipeline()
    mm.__dict__["qwen_texture"] = {"pipeline": pipe, "device": "cuda:0"}
    gen.loras = LoraManager(pipe, mm.config)
    with pytest.raises(RuntimeError, match="references still retain"):
        mm.release_qwen("qwen_texture", gen)


def test_calls_share_manager_lock():
    mm = ModelManager(config())
    active = 0
    peak = 0
    barrier = threading.Barrier(2)

    class ConcurrentGenerator(Generator):
        @qwen_stage("qwen_texture")
        def generate(self):
            nonlocal active, peak
            active += 1
            peak = max(peak, active)
            time.sleep(0.03)
            active -= 1

    def run():
        barrier.wait(timeout=5)
        ConcurrentGenerator(mm).generate()

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(run) for _ in range(2)]
        for future in futures:
            future.result(timeout=5)
    assert peak == 1


def test_dual_requires_hardware_and_true_80gb(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    with pytest.raises(ValueError, match="exactly two"):
        Config(model_lifecycle="dual_a100_80gb", vram_mode="80gb")
    with pytest.raises(ValueError, match="true 80gb"):
        Config(model_lifecycle="dual_a100_80gb", vram_mode="48gb")


def test_dual_shares_bundle_and_clears_adapters(monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(
        torch.cuda, "get_device_name", lambda i: "NVIDIA A100-SXM4-80GB"
    )
    cfg = Config(model_lifecycle="dual_a100_80gb", vram_mode="80gb")
    assert cfg.gpu_assignments["tile_generator"] == 1
    assert all(
        v == "cuda:1"
        for k, v in cfg.get_qwen_vram_config("cuda:1").items()
        if k.endswith("device")
    )
    mm = ModelManager(cfg)
    pipe = Pipeline()
    pipe.dit.vram_management_enabled = True
    pipe.clear_lora = lambda: None
    mm.__dict__["qwen_texture"] = {"pipeline": pipe, "device": "cuda:1"}
    assert mm.qwen_tiling is mm.qwen_texture
    manager = LoraManager(pipe, cfg)
    manager._loaded_paths = ("prior-texture",)
    mm._shared_loras = manager

    class Tiler:
        def __init__(self):
            self.mm = mm
            self.config = cfg

        @qwen_stage("qwen_tiling")
        def generate(self):
            assert manager.loaded_paths == ()
            return "tile"

    assert Tiler().generate() == "tile"
    assert mm.qwen_texture["pipeline"] is pipe
