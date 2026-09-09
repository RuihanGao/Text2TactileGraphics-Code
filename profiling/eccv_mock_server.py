"""Isolated localhost-only UI fixture; never used by the production launcher."""

import os
import time
from types import SimpleNamespace

import trimesh
import uvicorn
from PIL import Image

from text2tactilegraphics.ui.admission import Admission
from text2tactilegraphics.ui.prompt_planner import PromptPlan
from text2tactilegraphics.ui.public_app import create_server

mesh = "/tmp/eccv-mock.glb"
trimesh.creation.icosphere(subdivisions=2).export(mesh)


class LightweightPipeline:
    def generate(self, state, prompt):
        if not prompt.strip():
            raise ValueError("Please enter an object description.")
        state.status = "Understanding prompt"
        yield state
        state.update_plan(
            PromptPlan(shape_prompt="dolphin", regions=[], braille_label="dolphin")
        )
        state.base_image = Image.new("RGB", (64, 64), "white")
        state.status = "Generating object"
        yield state
        time.sleep(float(os.getenv("ECCV_MOCK_SECONDS", "5")))
        state.final_mesh = mesh
        state.status = "Done"
        yield state


service = SimpleNamespace(state="Ready", require_ready=lambda: None)
app = create_server(
    service, pipeline=LightweightPipeline(), admission=Admission.from_environment()
)
uvicorn.run(app, host="127.0.0.1", port=8083, access_log=False)
