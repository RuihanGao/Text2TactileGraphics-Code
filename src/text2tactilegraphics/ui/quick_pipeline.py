"""Session artifacts and resumable orchestration of the advanced UI handlers."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

from text2tactilegraphics.ui import handlers
from text2tactilegraphics.ui.prompt_planner import PromptPlan, PromptPlanner
from text2tactilegraphics.ui.state import AppState


class QuickSettings(BaseModel):
    base_steps: Literal[4, 40] = 4
    base_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    texture_steps: Literal[4, 40] = 4
    texture_seed: int = Field(default=42, ge=0, le=2**32 - 1)
    crop: bool = True
    tiling_method: Literal[
        "intra_tile_inpainting", "inter_tile_inpainting", "tiled_diffusion"
    ] = "intra_tile_inpainting"
    highpass: bool = True
    frequency: int = Field(default=120, ge=5, le=200)
    highpass_method: Literal["per_channel", "height_integration"] = "per_channel"
    normal_format: Literal["opengl", "directx"] = "opengl"
    displacement_mm: float = Field(default=5, ge=1, le=10)
    direction: Literal["normal", "z"] = "normal"
    repeats: int = Field(default=3, ge=1, le=10)
    plate_cm: float = Field(default=12, ge=5, le=30)
    flatten: bool = True
    thickness_mm: float = Field(default=4, ge=1, le=10)


@dataclass
class RegionArtifacts:
    mask: np.ndarray | None = None
    texture: Image.Image | None = None
    geometry: Image.Image | None = None
    tiled: Image.Image | None = None


@dataclass
class QuickState:
    original_prompt: str = ""
    plan: PromptPlan | None = None
    settings: QuickSettings = field(default_factory=QuickSettings)
    base_image: Image.Image | None = None
    regions: list[RegionArtifacts] = field(default_factory=list)
    final_mesh: str | None = None
    status: str = "Describe an object to begin."

    def invalidate(self, stage: str, index: int | None = None):
        """Only clear descendants: masks do not depend on texture, or vice versa."""
        self.final_mesh = None
        if stage == "base":
            self.base_image = None
        targets = self.regions if index is None else [self.regions[index]]
        for region in targets:
            if stage in ("base", "segmentation"):
                region.mask = None
            if stage == "texture":
                region.texture = None
            if stage in ("texture", "geometry"):
                region.geometry = None
            if stage in ("texture", "geometry", "tiling"):
                region.tiled = None

    def update_plan(self, plan: PromptPlan):
        plan = PromptPlan.model_validate(plan.model_dump())
        old = self.plan
        if old is None:
            self.regions = [RegionArtifacts() for _ in plan.regions]
        else:
            if old.shape_prompt != plan.shape_prompt:
                self.invalidate("base")
            # Match unchanged entries across insertions/removals/reordering first.
            remaining = list(zip(old.regions, self.regions))
            matched = {}
            for index, spec in enumerate(plan.regions):
                match = next(
                    (i for i, (p, _) in enumerate(remaining) if p == spec), None
                )
                if match is not None:
                    matched[index] = remaining.pop(match)[1]
            regions = []
            for index, spec in enumerate(plan.regions):
                if index in matched:
                    regions.append(matched[index])
                    continue
                match = next(
                    (
                        i
                        for i, (p, _) in enumerate(remaining)
                        if p.part_prompt == spec.part_prompt
                        or p.texture_prompt == spec.texture_prompt
                    ),
                    None,
                )
                if match is None:
                    regions.append(RegionArtifacts())
                    continue
                previous, artifacts = remaining.pop(match)
                if previous.part_prompt != spec.part_prompt:
                    artifacts.mask = None
                if previous.texture_prompt != spec.texture_prompt:
                    artifacts.texture = artifacts.geometry = artifacts.tiled = None
                regions.append(artifacts)
            self.regions = regions
        if old != plan:
            self.final_mesh = None
        self.plan = plan

    def update_settings(self, settings: QuickSettings):
        changed = {
            key
            for key, value in settings.model_dump().items()
            if getattr(self.settings, key) != value
        }
        if changed & {"base_steps", "base_seed"}:
            self.invalidate("base")
        if changed & {"texture_steps", "texture_seed"}:
            self.invalidate("texture")
        if "crop" in changed:
            self.invalidate("geometry")
        if changed & {"tiling_method", "highpass", "frequency", "highpass_method"}:
            self.invalidate("tiling")
        if changed:
            self.final_mesh = None
        self.settings = settings

    def replace_image(self, stage: str, image: Image.Image, index: int = 0):
        if image is None:
            raise ValueError("Upload an image first.")
        if stage == "segmentation":
            if self.base_image is None or image.size != self.base_image.size:
                raise ValueError(
                    "The mask must have the same dimensions as the base image."
                )
            mask = np.asarray(image.convert("L")) > 127
            if not mask.any():
                raise ValueError(
                    "The mask is empty. Use white for the selected region."
                )
        self.invalidate(stage, None if stage == "base" else index)
        if stage == "base":
            self.base_image = image.convert("RGB")
        elif stage == "segmentation":
            self.regions[index].mask = mask
        else:
            attr = {"texture": "texture", "geometry": "geometry", "tiling": "tiled"}[
                stage
            ]
            setattr(self.regions[index], attr, image.convert("RGB"))


class QuickPipeline:
    def __init__(self, planner: PromptPlanner, app_state: AppState | None = None):
        self.planner = planner
        self.app_state = app_state or AppState()

    def generate(self, state: QuickState, prompt: str):
        state.original_prompt = prompt
        state.status = "Understanding prompt"
        state.final_mesh = None
        yield state
        plan = self.planner.plan(prompt)
        state.update_plan(plan)
        yield from self.resume(state)

    def resume(self, state: QuickState):
        if state.plan is None:
            raise ValueError("Generate a plan first.")
        s, plan, app = state.settings, state.plan, self.app_state
        if state.base_image is None:
            state.status = "Generating object"
            yield state
            state.base_image = handlers.generate_base_image(
                plan.shape_prompt, "qwen_edit", s.base_steps, s.base_seed, app
            )
            yield state
        for i, (spec, region) in enumerate(zip(plan.regions, state.regions)):
            if region.mask is None:
                state.status = f"Finding texture regions · {i + 1}/{len(state.regions)}"
                yield state
                region.mask, _ = handlers.segment_with_text(
                    state.base_image, spec.part_prompt, app
                )
                yield state
        for i, (spec, region) in enumerate(zip(plan.regions, state.regions)):
            state.status = f"Creating tactile texture · {i + 1}/{len(state.regions)}"
            yield state
            if region.texture is None:
                region.texture = handlers.generate_texture_image(
                    spec.texture_prompt, s.texture_steps, s.texture_seed, app
                )
                yield state
            if region.geometry is None:
                _, region.geometry = handlers.generate_texture_geometry(
                    region.texture, s.crop, app
                )
                yield state
            if region.tiled is None:
                state.status = f"Making texture tileable · {i + 1}/{len(state.regions)}"
                yield state
                region.tiled = handlers.make_tileable(
                    region.geometry,
                    s.tiling_method,
                    100 if s.tiling_method == "tiled_diffusion" else 10,
                    42,
                    s.highpass,
                    s.frequency,
                    s.highpass_method,
                    app,
                )
                yield state
        if state.final_mesh is None:
            state.status = "Building tactile graphic and Braille"
            yield state
            segments = []
            for region in state.regions:
                handlers.save_segment(
                    state.base_image,
                    region.mask,
                    region.tiled,
                    s.displacement_mm / 1000,
                    s.direction,
                    s.repeats,
                    segments,
                )
            mesh = handlers.generate_final_mesh(
                state.base_image,
                segments,
                s.normal_format,
                "standard",
                plan.braille_label,
                s.plate_cm / 100,
                0.3,
                0.001,
                [],
                0.0005,
                s.flatten,
                s.thickness_mm / 1000,
            )
            if not isinstance(mesh, str):
                raise RuntimeError("The mesh handler did not return a file.")
            state.final_mesh = mesh
        state.status = "Done"
        yield state
