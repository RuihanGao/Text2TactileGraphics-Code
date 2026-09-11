"""Quick Trial contracts; slow tests use real handlers with explicit fixture plans."""

import json
import os
import time
from pathlib import Path
from unittest.mock import Mock

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from text2tactilegraphics.ui import quick_pipeline as qp
from text2tactilegraphics.ui.prompt_planner import GeminiPromptPlanner, PromptPlan
from text2tactilegraphics.ui.public_app import EXAMPLES, create_demo

PLANS = [
    {
        "shape_prompt": "a dolphin with wings",
        "regions": [
            {"part_prompt": "dolphin", "texture_prompt": "avocado skin texture"}
        ],
        "braille_label": "dolphin",
    },
    {
        "shape_prompt": "lamp",
        "regions": [
            {"part_prompt": "base of the lamp", "texture_prompt": "tree bark texture"},
            {"part_prompt": "lamp shade", "texture_prompt": "cloth_bag texture"},
        ],
        "braille_label": "lamp",
    },
]


class FixturePlanner:
    def plan(self, description):
        return PromptPlan.model_validate(PLANS[EXAMPLES.index(description)])


@pytest.fixture
def handlers(monkeypatch):
    result = {}
    image = Image.new("RGB", (16, 16), "white")
    returns = {
        "generate_base_image": image,
        "segment_with_text": (np.ones((16, 16), bool), image),
        "generate_texture_image": image,
        "generate_texture_geometry": (np.zeros((16, 16, 3)), image),
        "make_tileable": image,
        "generate_final_mesh": "/tmp/example.glb",
    }
    for name, value in returns.items():
        mock = Mock(return_value=value)
        monkeypatch.setattr(qp.handlers, name, mock)
        result[name] = mock
    return result


@pytest.mark.parametrize("index", [0, 1])
def test_full_orchestration(index, handlers):
    state = qp.QuickState()
    pipeline = qp.QuickPipeline(FixturePlanner())
    stages = [s.status for s in pipeline.generate(state, EXAMPLES[index])]
    assert stages[0] == "Understanding prompt" and stages[-1] == "Done"
    assert state.plan.model_dump() == PLANS[index]
    assert len(state.regions) == index + 1
    assert all(r.mask is not None and r.tiled is not None for r in state.regions)
    mesh_args = handlers["generate_final_mesh"].call_args.args
    assert len(mesh_args[1]) == index + 1
    assert mesh_args[4] == PLANS[index]["braille_label"]
    assert handlers["generate_base_image"].call_args.args[1:4] == ("qwen_edit", 4, 42)
    assert handlers["make_tileable"].call_args.args[2] == 10


def test_dependency_graph(handlers):
    state = qp.QuickState()
    pipeline = qp.QuickPipeline(FixturePlanner())
    list(pipeline.generate(state, EXAMPLES[1]))
    base = state.base_image
    textures = [r.texture for r in state.regions]
    plan = state.plan.model_copy(deep=True)
    plan.regions[0].part_prompt = "lamp foot"
    state.update_plan(plan)
    assert state.regions[0].mask is None
    assert state.regions[1].mask is not None
    assert [r.texture for r in state.regions] == textures
    list(pipeline.resume(state))
    assert handlers["segment_with_text"].call_count == 3
    assert handlers["generate_texture_image"].call_count == 2
    plan = state.plan.model_copy(deep=True)
    plan.regions[1].texture_prompt = "woven cloth"
    state.update_plan(plan)
    assert state.regions[1].mask is not None
    assert state.regions[1].texture is None
    list(pipeline.resume(state))
    assert handlers["generate_texture_image"].call_count == 3
    assert state.base_image is base
    plan.braille_label = "light"
    state.update_plan(plan)
    list(pipeline.resume(state))
    assert handlers["generate_base_image"].call_count == 1
    assert handlers["generate_texture_image"].call_count == 3
    assert handlers["generate_final_mesh"].call_count == 4
    state.invalidate("base")
    assert all(r.mask is None and r.tiled is not None for r in state.regions)


def test_settings_upload_and_reordering(handlers):
    state = qp.QuickState()
    pipeline = qp.QuickPipeline(FixturePlanner())
    list(pipeline.generate(state, EXAMPLES[1]))
    before = state.regions[:]
    plan = state.plan.model_copy(deep=True)
    plan.regions.reverse()
    state.update_plan(plan)
    assert state.regions[0] is before[1]
    state.update_settings(qp.QuickSettings(frequency=100))
    assert all(r.tiled is None and r.geometry is not None for r in state.regions)
    list(pipeline.resume(state))
    state.replace_image("texture", Image.new("RGB", (16, 16)), 0)
    assert state.regions[0].geometry is None
    assert state.regions[0].mask is not None
    assert state.regions[1].tiled is not None
    with pytest.raises(ValueError, match="dimensions"):
        state.replace_image("segmentation", Image.new("L", (4, 4)), 0)
    with pytest.raises(ValueError, match="empty"):
        state.replace_image("segmentation", Image.new("L", (16, 16)), 0)


def test_failure_keeps_completed_stages(handlers):
    pipeline = qp.QuickPipeline(FixturePlanner())
    state = qp.QuickState()
    handlers["make_tileable"].side_effect = RuntimeError("test failure")
    with pytest.raises(RuntimeError):
        list(pipeline.generate(state, EXAMPLES[0]))
    assert state.final_mesh is None and state.regions[0].geometry is not None
    handlers["make_tileable"].side_effect = None
    list(pipeline.resume(state))
    assert state.status == "Done"
    assert handlers["generate_texture_image"].call_count == 1


def test_schema_and_missing_credentials(monkeypatch):
    for name in ("GEMINI_API_KEY", "GENAI_API_KEY"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(RuntimeError, match="GEMINI_API_KEY"):
        GeminiPromptPlanner().plan(EXAMPLES[0])
    with pytest.raises(ValueError):
        GeminiPromptPlanner().plan(" ")
    with pytest.raises(ValidationError):
        PromptPlan.model_validate({**PLANS[0], "shape_prompt": " "})
    with pytest.raises(ValidationError):
        PromptPlan.model_validate({**PLANS[0], "regions": [{"part_prompt": "dolphin"}]})
    with pytest.raises(ValidationError):
        PromptPlan.model_validate({**PLANS[0], "braille_label": "🐬"})


@pytest.mark.parametrize("index", [0, 1])
def test_gemini_schema_contract(index, monkeypatch):
    from google import genai

    monkeypatch.setenv("GEMINI_API_KEY", "unit-test-placeholder")
    client = Mock()
    client.__enter__ = Mock(return_value=client)
    client.__exit__ = Mock(return_value=False)
    client.models.generate_content.side_effect = [
        Mock(text='{"safe":true}'),
        Mock(text=json.dumps(PLANS[index])),
    ]
    monkeypatch.setattr(genai, "Client", Mock(return_value=client))
    assert GeminiPromptPlanner().plan(EXAMPLES[index]).model_dump() == PLANS[index]
    config = client.models.generate_content.call_args.kwargs["config"]
    assert config.response_schema is None
    assert config.response_json_schema == PromptPlan.model_json_schema()
    assert config.response_json_schema["additionalProperties"] is False
    assert (
        config.response_json_schema["$defs"]["TextureRegion"]["additionalProperties"]
        is False
    )
    # Provider output still passes strict local validation, even if it ignores
    # JSON Schema constraints or invents properties.
    client.models.generate_content.side_effect = [
        Mock(text='{"safe":true}'),
        Mock(text=json.dumps({**PLANS[index], "unexpected": "field"})),
    ]
    with pytest.raises(RuntimeError, match="Prompt checking or planning failed"):
        GeminiPromptPlanner().plan(EXAMPLES[index])
    client.models.generate_content.side_effect = [
        Mock(text='{"safe":true}'),
        Mock(text='{"shape_prompt": "lamp"}'),
    ]
    with pytest.raises(RuntimeError, match="Prompt checking or planning failed"):
        GeminiPromptPlanner().plan(EXAMPLES[index])


def test_ui_layout_and_queue():
    demo = create_demo(planner=FixturePlanner())
    components = demo.config["components"]
    accordion = next(c for c in components if c["type"] == "accordion")
    assert accordion["props"]["label"] == "Customize intermediate steps"
    assert accordion["props"]["open"] is False
    assert not any(c["type"] == "row" for c in components)
    assert (
        len(
            {
                fn.concurrency_id
                for fn in demo.fns.values()
                if fn.fn and fn.fn.__name__ in {"start", "edit", "region_images"}
            }
        )
        == 1
    )


@pytest.mark.slow
@pytest.mark.parametrize("index", [0, 1])
def test_real_quick_workflow(index):
    """Real GPU path with supplied expected plan, not a live planner validation."""
    import torch
    import trimesh

    if not torch.cuda.is_available():
        pytest.skip("Requires GPU and released model caches")
    root = Path("output-quick-trial") / str(index)
    root.mkdir(parents=True, exist_ok=True)
    state = qp.QuickState()
    pipeline = qp.QuickPipeline(FixturePlanner())
    events = []
    started = time.perf_counter()
    try:
        for result in pipeline.generate(state, EXAMPLES[index]):
            events.append(
                {
                    "stage": result.status,
                    "elapsed_seconds": time.perf_counter() - started,
                }
            )
        assert state.final_mesh and Path(state.final_mesh).is_file()
        mesh = trimesh.load(state.final_mesh, force="mesh")
        assert len(mesh.vertices) and len(mesh.faces)
        (root / "final.glb").write_bytes(Path(state.final_mesh).read_bytes())
        state.base_image.save(root / "base.png")
        for i, region in enumerate(state.regions):
            Image.fromarray((region.mask * 255).astype("uint8")).save(
                root / f"mask-{i}.png"
            )
            region.texture.save(root / f"texture-{i}.png")
            region.tiled.save(root / f"tiled-{i}.png")
        details = {
            "plan": state.plan.model_dump(),
            "planner": "fixture (no live credentials)",
            "lifecycle": os.getenv("TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE"),
            "seconds": time.perf_counter() - started,
            "vertices": len(mesh.vertices),
            "faces": len(mesh.faces),
            "watertight": mesh.is_watertight,
            "events": events,
        }
        (root / "result.json").write_text(json.dumps(details, indent=2))
    finally:
        (root / "events.json").write_text(json.dumps(events, indent=2))


def test_textureless_object_and_isolated_sessions(handlers):
    first = qp.QuickState()
    first.update_plan(PromptPlan(shape_prompt="lamp", regions=[], braille_label="lamp"))
    pipeline = qp.QuickPipeline(FixturePlanner())
    list(pipeline.resume(first))
    assert first.final_mesh
    assert handlers["segment_with_text"].call_count == 0
    assert handlers["generate_texture_image"].call_count == 0
    second = qp.QuickState()
    assert second.plan is None and second.final_mesh is None
    assert second.regions is not first.regions


@pytest.mark.slow
@pytest.mark.parametrize("index", [0, 1])
def test_live_prompt_planner(index):
    if not (os.getenv("GEMINI_API_KEY") or os.getenv("GENAI_API_KEY")):
        pytest.skip("Live planner requires a server-side Gemini key")
    plan = GeminiPromptPlanner().plan(EXAMPLES[index])
    expected = PLANS[index]
    assert expected["braille_label"] in plan.braille_label.lower()
    assert len(plan.regions) == len(expected["regions"])
    assert expected["braille_label"] in plan.shape_prompt.lower()
    if index == 0:
        assert "wings" in plan.shape_prompt.lower()
        assert "avocado" in plan.regions[0].texture_prompt.lower()
        assert "avocado" not in plan.shape_prompt.lower()
    else:
        assert any(
            "bark" in r.texture_prompt.lower() and "base" in r.part_prompt.lower()
            for r in plan.regions
        )
        assert any(
            "cloth" in r.texture_prompt.lower() and "shade" in r.part_prompt.lower()
            for r in plan.regions
        )


def test_inserted_region_does_not_steal_unchanged_artifacts(handlers):
    state = qp.QuickState()
    pipeline = qp.QuickPipeline(FixturePlanner())
    list(pipeline.generate(state, EXAMPLES[0]))
    original = state.regions[0]
    plan = state.plan.model_copy(deep=True)
    plan.regions.insert(0, plan.regions[0].model_copy(update={"part_prompt": "wings"}))
    state.update_plan(plan)
    assert state.regions[1] is original
    assert original.mask is not None and original.tiled is not None
    assert state.regions[0].mask is None


def test_new_request_clears_stale_editor_fields():
    demo = create_demo(planner=FixturePlanner())
    start = next(
        fn.fn for fn in demo.fns.values() if fn.fn and fn.fn.__name__ == "start"
    )
    update = next(start(EXAMPLES[0]))
    values = {
        getattr(component, "label", None): value for component, value in update.items()
    }
    assert values["Shape prompt"] == ""
    assert values["Braille label"] == ""
    assert values["Texture regions (add or remove rows)"] == []
