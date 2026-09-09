# Safe to edit

**SAFE FOR UI / AESTHETIC EDITS**, scoped to presentation:

- `src/text2tactilegraphics/ui/public_app.py`: `CSS`, explanatory copy,
  component labels/layout in `create_demo`, example presentation (`EXAMPLES`,
  `gr.Examples`), viewer/gallery sizing and accordion appearance. Keep component
  identity and settings/default values unchanged. Existing example strings are
  used by tests; coordinate changes to their text.
- `src/text2tactilegraphics/ui/theme.py`: colors, typography, spacing, radii and
  button styles for the **advanced** UI. Quick Trial does not use this theme;
  edit its `CSS` in `public_app.py` instead.

There is no separate Quick Trial stylesheet/static-asset module. Existing
`src/text2tactilegraphics/assets/` files include inference inputs; do not replace
those to change branding. Keep progress text accessible and visible.

# Avoid changing

**DO NOT MODIFY FOR VISUAL-ONLY CHANGES:**

- `src/text2tactilegraphics/config.py`: lifecycle, GPU assignment, model/checkpoint configuration.
- `src/text2tactilegraphics/generation/models.py`: model residency, Qwen loading,
  `LoraManager`, adapter switching and lifecycle decorators.
- `src/text2tactilegraphics/generation/base_image_generation.py` and
  `src/text2tactilegraphics/generation/texture_generation.py`: inference behavior.
- `src/text2tactilegraphics/ui/inference_service.py`: warmup/readiness.
- `src/text2tactilegraphics/ui/admission.py`: concurrency, queue and ownership.
- `src/text2tactilegraphics/ui/quick_pipeline.py`,
  `src/text2tactilegraphics/ui/prompt_planner.py`,
  `src/text2tactilegraphics/ui/handlers.py`,
  `src/text2tactilegraphics/ui/state.py`: planning, state and generation.
- `src/text2tactilegraphics/ui/public_app.py`: callbacks (`readiness`, `snapshot`,
  `stream`, `start`, `edit`, `reserve`, `bind`), callback wiring, status escaping,
  queue, cookies and `create_server` health routes. **This is a mixed UI/backend file.**
- `src/text2tactilegraphics/ui/app.py`: advanced workflow and runtime controls.
- `src/text2tactilegraphics/geometry/`: mesh, displacement and Braille algorithms.
- `launch_eccv.sh`, `scripts/stop_eccv.py`, `pyproject.toml`, `uv.lock`:
  production lifecycle and reproducible environment.

# Before opening a PR

- Work in a separate checkout and styling branch; leave the live service untouched.
- Preview with `.venv/bin/python profiling/eccv_mock_server.py` on localhost:8083.
  It uses mock images/mesh, not GPU inference; Customize resume is unsupported.
- Run `.venv/bin/python -m pytest tests/ui/test_quick_trial.py -q`.
- Run `.venv/bin/python -m pytest -q tests`.
- Run `.venv/bin/ruff check` and `.venv/bin/ruff format --check` with the paths of
  every changed Python file, plus `git diff --check`.
- Verify mobile layout (390×844), readable text and usable viewer/download sizing.
- Verify Advanced/Customize remains collapsed by default.
- Verify Generate, Busy, Ready and progress/error states remain visible.
- Inspect `git diff --name-only` and `git diff`: no backend sections, inference
  defaults, checkpoints, queue policies or dependency files should change unexpectedly.
- Submit a PR; do not push styling changes directly to the deployment branch.

See [ECCV_DEMO.md](ECCV_DEMO.md) for architecture, validation commands and operations.
