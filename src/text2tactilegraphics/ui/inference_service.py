"""Explicit startup readiness for the resident poster lifecycle."""

import logging
import threading
import time

import torch

from text2tactilegraphics.generation.models import global_model_manager
from text2tactilegraphics.ui.prompt_planner import PromptPlan
from text2tactilegraphics.ui.quick_pipeline import QuickPipeline, QuickState

logger = logging.getLogger(__name__)


class InferenceService:
    def __init__(self):
        self.state = "Warming"
        self.startup_seconds = None
        self.error_class = None
        self.resident_gib = []
        self._lock = threading.Lock()

    def warmup(self):
        with self._lock:
            if self.state == "Ready":
                return
            start = time.perf_counter()
            try:
                mm = global_model_manager()
                if mm.config.model_lifecycle == "dual_a100_80gb":
                    mm.warmup()

                class WarmupPlanner:
                    def plan(self, _):
                        return PromptPlan(
                            shape_prompt="a dolphin with wings",
                            regions=[
                                {
                                    "part_prompt": "dolphin",
                                    "texture_prompt": "an avocado skin",
                                }
                            ],
                            braille_label="dolphin",
                        )

                # Exercise released kernels and geometry lazy setup before acceptance.
                pipeline = QuickPipeline(WarmupPlanner())
                for _ in pipeline.generate(QuickState(), "startup warmup"):
                    pass
                for i in range(torch.cuda.device_count()):
                    torch.cuda.synchronize(i)
                self.resident_gib = [
                    torch.cuda.memory_allocated(i) / 1024**3
                    for i in range(torch.cuda.device_count())
                ]
                self.startup_seconds = time.perf_counter() - start
                self.state = "Ready"
                logger.info("Inference Ready after %.2f seconds", self.startup_seconds)
            except Exception as exc:
                self.error_class = type(exc).__name__
                self.state = "Unhealthy"
                logger.error("Inference warmup failed (%s)", self.error_class)

    def start(self):
        threading.Thread(
            target=self.warmup, name="inference-warmup", daemon=True
        ).start()

    def require_ready(self):
        if self.state != "Ready":
            raise ValueError(
                "The live demo is warming up or temporarily unavailable. Please try again shortly."
            )
