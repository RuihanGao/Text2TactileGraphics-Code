# Quick Trial

Launch the separate mobile interface using the existing project environment:

```bash
cd /workspace/Text2TactileGraphics-Code
./launch_quick_trial.sh
```

The launcher privately loads `/workspace/private/gemini.env`, requires a nonempty
`GEMINI_API_KEY`, selects `single_a100_80gb`, and binds explicitly to
`0.0.0.0:8080`. It preserves inherited model caches and requires no `uv` command
in your shell. `QUICK_TRIAL_KEY_FILE`, `GRADIO_SERVER_NAME`, and
`GRADIO_SERVER_PORT` can override the credential path and address. It fails if
the port is occupied; it never stops another service. Keep the launch process
running. Saving a new key requires restarting **only Quick Trial**.

On the validated pod, nginx port **8081 forwards to Quick Trial on 8080**.
Use a RunPod HTTP connection exposing port 8081 (or expose 8080 directly).
External access was not verified: the pod-ID-based 8081 URL returned 404 and
8080 returned 403 from this environment. Local proxy access was tested.
Port 7861 forwards to the advanced
app on 7860; that is a different UI. Verify the page title is **Quick Trial**.
These proxy mappings are pod configuration, not a guarantee for other pods.
For a fresh environment, reproduce `uv.lock` with `uv sync --frozen` first;
do not rebuild the functioning environment just because `uv` is absent from PATH.

One prompt and **Generate** run planning, object generation, all segmentations,
per-region texture generation, normal estimation, tiling, and the final geometry
handler including standard Braille and GLB export. The base image appears as soon
as it is ready. Stage text and intermediate region images update throughout.
Download returns the same GLB displayed in the viewer.

## Planner

`PromptPlanner` is a provider protocol; inject another implementation into
`create_demo(planner=...)` or `QuickPipeline`. The default `GeminiPromptPlanner`
uses the already-installed Google GenAI SDK, a text-only request, JSON structured
output via `response_json_schema`, temperature zero, and strict local Pydantic
validation. The legacy `response_schema` conversion was rejected by the live
API for `additional_properties`; local extra-field and length limits remain
intact. It loads no image model for parsing. `TEXT2TACTILEGRAPHICS_PLANNER_MODEL` overrides `gemini-3.6-flash`.
Only the typed description is sent to the planner service. Credentials remain
server-side. Missing credentials and invalid/provider responses stop planning
with an actionable error; there is no silent heuristic fallback.

Plans allow zero to eight texture regions, descriptions up to 2000 characters,
and a short ASCII English Braille label up to 40 characters. The examples in the
planner instruction distinguish shape features such as wings from textures,
and preserve separate lamp-base and lamp-shade assignments.

## Customize and resume

The collapsed **Customize intermediate steps** accordion edits the very same
`QuickState` used by the default flow. Edit shape, region rows, Braille, or
settings, then select **Apply edits and continue**. Empty region rows are ignored;
partially filled rows are rejected. Use the region selector to inspect each
mask, texture, normal map, and tileable map. All regions also appear in a gallery.
Upload a replacement and use its corresponding **Use … and continue** button.
Masks must match the base dimensions; white selects the region. Apply prompt
edits before uploading replacements. Use **Rerun this stage and continue** to
explicitly regenerate a selected stage even if its inputs have not changed.

| Changed input | Invalidated artifacts |
| --- | --- |
| Shape, base seed/steps, base image | Base (unless uploaded), all masks, final mesh |
| Region segmentation prompt/mask | That mask (unless uploaded), final mesh |
| Region texture prompt/image | That texture (unless uploaded), geometry, tiling, final mesh |
| Crop/geometry image | Geometry (unless uploaded), tiling, final mesh |
| Tiling/filter settings/map | Tiling (unless uploaded), final mesh |
| Braille, displacement, normal convention, plate settings | Final mesh |

Unchanged regions retain their artifacts when reordered or removed. Textures
are independent of the base image, so changing the object retains them.
Changing a global texture/tiling setting affects every region. Final mesh
rebuilds rerun the existing geometry handler, including its base-depth and
plate segmentation work; this UI does not cache inside or rewrite that handler.
Failed runs retain completed artifacts for resume. A new **Generate** starts a
fresh artifact graph. Session state is isolated; GPU callbacks share a serial
queue (maximum 16 pending requests). Gradio copies served outputs into its cache
and cleans that cache after a day. The existing core handler also writes source
GLBs to the system temporary directory; operators should retain their usual
server temporary-file cleanup policy.

## Backend independence

The frontend does not read lifecycle environment variables, assign GPUs,
quantize models, change checkpoints, or unload models. It calls the existing
`AppState`/handlers; their generator decorators apply the backend lifecycle.
The defaults match the advanced UI: Qwen edit, seed 42, 4-step base and texture,
10-step intra-tile inpainting, normal geometry, center crop, high-pass frequency
120, 5 mm displacement, three repeats, and a 12 cm plate with standard Braille.
Tiled Diffusion, if selected, uses the advanced UI's 100-step setting.

This branch's backend currently accepts only `cached` and `single_a100_80gb`.
It rejects `dual_a100_80gb` during configuration. Once the backend implements
that mode, this frontend requires no changes; dual-GPU execution cannot be
validated on the current backend. No lifecycle implementation is changed here.

## Validation

```bash
uv run --frozen pytest tests/ui/test_quick_trial.py -q
TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb \
  uv run --frozen pytest tests/ui/test_quick_trial.py -m slow -q
```

The slow tests execute real generation handlers for both requested examples,
using explicitly supplied expected plans. They save per-stage timestamps,
plans, images, masks, and mesh results under ignored `output-quick-trial/`.
They are not live planner tests. Live planner verification requires a server
Gemini key; unit tests exercise the provider request/schema using mocked SDK
responses. See `QUICK_TRIAL_VALIDATION.md` for results and limitations.
