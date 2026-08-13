"""Application state for the Gradio UI."""

import logging
from collections.abc import Callable
from datetime import datetime
from functools import cached_property
from pathlib import Path
from typing import Literal

import numpy as np
from PIL import Image

from text2tactilegraphics import TiledDiffusion
from text2tactilegraphics.config import SENSOR_TILED_DIR, global_config
from text2tactilegraphics.generation.base_image_generation import BaseImageGenerator
from text2tactilegraphics.generation.segmentation import SegmentationEngine
from text2tactilegraphics.generation.texture_generation import (
    GeometryEstimator,
    TextureGenerator,
)
from text2tactilegraphics.generation.tileable_patch_generation import (
    InterTilePatchGenerator,
    IntraTilePatchGenerator,
)

logger = logging.getLogger(__name__)

TilingMethod = Literal[
    "intra_tile_inpainting", "inter_tile_inpainting", "tiled_diffusion"
]

# Default location for intermediate-result dumps
_DEFAULT_OUTPUT_DIR = Path(__file__).parent / "output"


def get_sensor_image_paths() -> list[str]:
    """Filesystem paths to bundled sensor-based tiled normal maps."""
    if not SENSOR_TILED_DIR.exists():
        return []
    return sorted(
        str(f) for f in SENSOR_TILED_DIR.glob("*.png") if not f.name.startswith(".")
    )


class AppState:
    def __init__(self) -> None:
        self.config = global_config()

        # Intermediate-results saving (off by default).
        self.save_intermediate_results: bool = False
        self.output_dir: str = str(_DEFAULT_OUTPUT_DIR)
        self.session_id: str | None = None

    # ----------------------------------------------------------------- lazy loaders

    def get_seg_engine(self) -> SegmentationEngine:
        return self._seg_engine

    @cached_property
    def _seg_engine(self) -> SegmentationEngine:
        return SegmentationEngine()

    def get_texture_gen(self) -> TextureGenerator:
        return self._texture_gen

    @cached_property
    def _texture_gen(self) -> TextureGenerator:
        return TextureGenerator()

    def get_geom_estimator(self) -> GeometryEstimator:
        return self._geom_estimator

    @cached_property
    def _geom_estimator(self) -> GeometryEstimator:
        return GeometryEstimator()

    def get_base_gen(self) -> BaseImageGenerator:
        return self._base_gen

    @cached_property
    def _base_gen(self) -> BaseImageGenerator:
        return BaseImageGenerator()

    def get_tiling_gen(
        self, method: TilingMethod
    ) -> IntraTilePatchGenerator | InterTilePatchGenerator | TiledDiffusion:
        return self._tiling_gens[method]

    @cached_property
    def _tiling_gens(
        self,
    ) -> dict[
        TilingMethod,
        IntraTilePatchGenerator | InterTilePatchGenerator | TiledDiffusion,
    ]:
        return {
            "intra_tile_inpainting": IntraTilePatchGenerator(),
            "inter_tile_inpainting": InterTilePatchGenerator(),
            "tiled_diffusion": TiledDiffusion(),
        }

    # ----------------------------------------------------------------- intermediate-results saving

    def _get_save_path(self, filename: str) -> Path:
        """Resolve `filename` under <output_dir>/<session_id>/, creating dirs."""
        if self.session_id is None:
            self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = self.session_id
        assert session_id is not None
        save_dir = Path(self.output_dir) / session_id
        save_dir.mkdir(parents=True, exist_ok=True)
        return save_dir / filename

    def _save(self, filename: str, writer: Callable[[Path], None]) -> str:
        """Run `writer(path)` under the session output dir, swallowing errors.

        Returns the saved path (and logs it) on success; returns "" when
        saving is disabled or fails.
        """
        if not self.save_intermediate_results:
            return ""
        try:
            path = self._get_save_path(filename)
            writer(path)
        except Exception as e:  # noqa: BLE001 — save failures are non-fatal
            logger.warning("Failed to save %s: %s", filename, e)
            return ""
        logger.info("Saved %s to: %s", filename, path)
        return str(path)

    def save_image(self, image: Image.Image, filename: str) -> str:
        """Save `image` under the session output dir; "" on disable/error."""
        return self._save(filename, lambda p: image.save(p))

    def save_array(self, array: np.ndarray, filename: str) -> str:
        """Save `array` (`.npy`) under the session output dir."""
        return self._save(filename, lambda p: np.save(p, array))
