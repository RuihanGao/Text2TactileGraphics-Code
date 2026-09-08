"""Mobile Quick Trial; run with python -m text2tactilegraphics.ui.public_app."""

from html import escape

import gradio as gr
from pydantic import ValidationError

from text2tactilegraphics.generation.utils import mask_to_image
from text2tactilegraphics.ui.prompt_planner import (
    GeminiPromptPlanner,
    PlannerError,
    PromptPlan,
)
from text2tactilegraphics.ui.quick_pipeline import (
    QuickPipeline,
    QuickSettings,
    QuickState,
)

EXAMPLES = [
    "a dolphin with wings with an avocado skin texture.",
    "lamp, the base of the lamp has a tree bark texture, and the lamp shade has a cloth_bag texture.",
    "a butterfly with a woven fabric texture on its wings",
]
CSS = """
.gradio-container {max-width: 760px !important; margin: auto;}
button {min-height: 48px !important; touch-action: manipulation;}
textarea, input {font-size: 16px !important;}
#quick-status {min-height: 48px;}
@media (max-width: 600px) {.gradio-container {padding: 12px !important;}}
"""


def status_html(message):
    return f'<div role="status" aria-live="polite">{escape(message)}</div>'


def create_demo(planner=None, pipeline=None):
    pipeline = pipeline or QuickPipeline(planner or GeminiPromptPlanner())
    with gr.Blocks(
        title="Text2TactileGraphics · Quick Trial", delete_cache=(3600, 86400)
    ) as demo:
        state = gr.State(QuickState())
        gr.Markdown(
            "# Quick Trial\nDescribe an object and its textures. We’ll turn it into a tactile graphic."
        )
        prompt = gr.Textbox(
            label="What would you like to create?",
            placeholder="A dolphin with wings and avocado skin texture…",
            lines=4,
            max_length=2000,
        )
        gr.Examples(
            [[text] for text in EXAMPLES], inputs=prompt, label="Try an example"
        )
        generate = gr.Button("Generate", variant="primary", size="lg")
        status = gr.HTML(
            status_html("Describe an object to begin."),
            elem_id="quick-status",
            container=True,
        )
        preview = gr.Image(
            label="Your object", type="pil", interactive=False, visible=False
        )
        mesh = gr.Model3D(label="Your tactile graphic", height=360)
        download = gr.DownloadButton("Download mesh", interactive=False, size="lg")
        with gr.Accordion("Customize intermediate steps", open=False):
            shape = gr.Textbox(label="Shape prompt")
            regions = gr.Dataframe(
                headers=["Segmentation prompt", "Texture prompt"],
                datatype=["str", "str"],
                type="array",
                column_count=2,
                row_count=0,
                label="Texture regions (add or remove rows)",
                interactive=True,
            )
            braille = gr.Textbox(label="Braille label", max_length=40)
            gr.Markdown(
                "Apply edits to rebuild only affected stages. Upload replacements using the buttons below. White mask pixels select the region; black pixels exclude it."
            )
            settings = {}
            settings["base_steps"] = gr.Radio([4, 40], value=4, label="Object steps")
            settings["base_seed"] = gr.Number(
                value=42, precision=0, label="Object seed"
            )
            settings["texture_steps"] = gr.Radio(
                [4, 40], value=4, label="Texture steps"
            )
            settings["texture_seed"] = gr.Number(
                value=42, precision=0, label="Texture seed"
            )
            settings["crop"] = gr.Checkbox(
                value=True, label="Center crop texture geometry"
            )
            settings["tiling_method"] = gr.Dropdown(
                ["intra_tile_inpainting", "inter_tile_inpainting", "tiled_diffusion"],
                value="intra_tile_inpainting",
                label="Tiling method",
            )
            settings["highpass"] = gr.Checkbox(value=True, label="High-pass filter")
            settings["frequency"] = gr.Slider(
                5, 200, value=120, step=5, label="High-pass frequency"
            )
            settings["highpass_method"] = gr.Radio(
                ["per_channel", "height_integration"],
                value="per_channel",
                label="High-pass method",
            )
            settings["normal_format"] = gr.Radio(
                ["opengl", "directx"], value="opengl", label="Normal map convention"
            )
            settings["displacement_mm"] = gr.Slider(
                1, 10, value=5, step=0.1, label="Texture depth (mm)"
            )
            settings["direction"] = gr.Radio(
                ["normal", "z"], value="normal", label="Displacement direction"
            )
            settings["repeats"] = gr.Slider(
                1, 10, value=3, step=1, label="Tile repeats"
            )
            settings["plate_cm"] = gr.Slider(5, 30, value=12, label="Plate size (cm)")
            settings["flatten"] = gr.Checkbox(value=True, label="Flatten plate")
            settings["thickness_mm"] = gr.Slider(
                1, 10, value=4, step=0.1, label="Plate thickness (mm)"
            )
            apply = gr.Button("Apply edits and continue", size="lg")
            base = gr.Image(
                label="Base image", type="pil", sources=["upload"], interactive=True
            )
            use_base = gr.Button("Use this base image and continue")
            selected = gr.Dropdown(
                choices=[], type="index", label="Region to customize"
            )
            mask = gr.Image(
                label="Segmentation mask",
                type="pil",
                sources=["upload"],
                interactive=True,
            )
            use_mask = gr.Button("Use this mask and continue")
            texture = gr.Image(
                label="Texture output", type="pil", sources=["upload"], interactive=True
            )
            use_texture = gr.Button("Use this texture and continue")
            geometry = gr.Image(
                label="Texture geometry (normal map)",
                type="pil",
                sources=["upload"],
                interactive=True,
            )
            use_geometry = gr.Button("Use this geometry and continue")
            tiled = gr.Image(
                label="Tileable output",
                type="pil",
                sources=["upload"],
                interactive=True,
            )
            use_tiled = gr.Button("Use this tileable output and continue")
            gallery = gr.Gallery(
                label="All regions: mask, texture, tileable output",
                columns=1,
                height=400,
            )
            stage = gr.Dropdown(
                ["base", "segmentation", "texture", "geometry", "tiling", "mesh"],
                value="segmentation",
                label="Regenerate stage (selected region)",
            )
            rerun = gr.Button("Rerun this stage and continue")

        def region_images(s, index):
            if index is None or not 0 <= index < len(s.regions):
                return None, None, None, None
            r = s.regions[index]
            return (
                None if r.mask is None else mask_to_image(r.mask),
                r.texture,
                r.geometry,
                r.tiled,
            )

        def snapshot(s, index):
            index = min(index or 0, len(s.regions) - 1) if s.regions else None
            plan = s.plan
            images = []
            for i in range(len(s.regions)):
                m, t, _, tile = region_images(s, i)
                for label, img in [("mask", m), ("texture", t), ("tileable", tile)]:
                    if img is not None:
                        images.append((img, f"Region {i + 1}: {label}"))
            result = {
                state: s,
                status: status_html(s.status),
                preview: gr.Image(value=s.base_image, visible=s.base_image is not None),
                mesh: s.final_mesh,
                download: gr.DownloadButton(
                    value=s.final_mesh, interactive=s.final_mesh is not None
                ),
                base: s.base_image,
                gallery: images,
            }
            if plan is not None:
                names = [
                    f"{i + 1}. {r.part_prompt}" for i, r in enumerate(plan.regions)
                ]
                result.update(
                    {
                        shape: plan.shape_prompt,
                        regions: [
                            [r.part_prompt, r.texture_prompt] for r in plan.regions
                        ],
                        braille: plan.braille_label,
                        selected: gr.Dropdown(
                            choices=names,
                            value=names[index] if index is not None else None,
                        ),
                    }
                )
            else:
                result.update(
                    {
                        shape: "",
                        regions: [],
                        braille: "",
                        selected: gr.Dropdown(choices=[], value=None),
                    }
                )
            result.update(
                dict(zip([mask, texture, geometry, tiled], region_images(s, index)))
            )
            result.update(
                {
                    component: getattr(s.settings, name)
                    for name, component in settings.items()
                }
            )
            return result

        outputs = [
            state,
            status,
            preview,
            mesh,
            download,
            base,
            gallery,
            shape,
            regions,
            braille,
            selected,
            mask,
            texture,
            geometry,
            tiled,
            *settings.values(),
        ]

        def stream(s, index, iterator):
            try:
                for current in iterator:
                    yield snapshot(current, index)
            except Exception as exc:
                failed_stage = s.status
                s.final_mesh = None
                if isinstance(exc, PlannerError):
                    detail = str(exc)
                elif isinstance(exc, ValidationError):
                    detail = "Check the prompt fields and parameter limits, then apply your edits again."
                elif isinstance(exc, ValueError):
                    detail = str(exc)
                else:
                    detail = "Check the server configuration or edit the relevant stage and retry."
                s.status = f"Could not finish: {failed_stage}. {detail} Completed stages are saved."
                yield snapshot(s, index)
                gr.Warning(
                    "Generation stopped. Completed intermediate results are available below."
                )

        def start(text):
            # A new Generate request starts an isolated session artifact graph.
            s = QuickState()
            yield from stream(s, 0, pipeline.generate(s, text))

        def edit(s, index, shape_text, rows, label, action, image, *values):
            def work():
                s.status = "Applying edits"
                if s.plan is None:
                    raise ValueError("Generate a plan first.")
                if any(len(row) != 2 for row in (rows or [])):
                    raise ValueError(
                        "Keep exactly two columns: segmentation prompt and texture prompt."
                    )
                plan = PromptPlan.model_validate(
                    {
                        "shape_prompt": shape_text,
                        "regions": [
                            {"part_prompt": row[0], "texture_prompt": row[1]}
                            for row in (rows or [])
                            if any(str(cell).strip() for cell in row)
                        ],
                        "braille_label": label,
                    }
                )
                new_settings = QuickSettings.model_validate(dict(zip(settings, values)))
                if action.startswith("upload:") and plan != s.plan:
                    raise ValueError(
                        "Apply prompt edits before uploading a replacement."
                    )
                s.update_plan(plan)
                s.update_settings(new_settings)
                if action.startswith("upload:"):
                    kind = action.split(":", 1)[1]
                    if kind != "base" and (index is None or index >= len(s.regions)):
                        raise ValueError("Select a region first.")
                    s.replace_image(kind, image, index or 0)
                elif action != "apply":
                    if action not in ("base", "mesh") and (
                        index is None or index >= len(s.regions)
                    ):
                        raise ValueError("Select a region first.")
                    s.invalidate(action, None if action in ("base", "mesh") else index)
                yield from pipeline.resume(s)

            yield from stream(s, index, work())

        queue = {
            "concurrency_id": "quick_trial",
            "concurrency_limit": 1,
            "api_visibility": "private",
            "trigger_mode": "once",
        }
        generate.click(start, inputs=prompt, outputs=outputs, **queue)
        inputs = [state, selected, shape, regions, braille]
        # Every GPU action uses the same queue across sessions and stages.
        apply.click(
            edit,
            inputs=inputs
            + [gr.State("apply"), gr.State(None)]
            + list(settings.values()),
            outputs=outputs,
            **queue,
        )
        rerun.click(
            edit,
            inputs=inputs + [stage, gr.State(None)] + list(settings.values()),
            outputs=outputs,
            **queue,
        )
        for button, kind, component in [
            (use_base, "base", base),
            (use_mask, "segmentation", mask),
            (use_texture, "texture", texture),
            (use_geometry, "geometry", geometry),
            (use_tiled, "tiling", tiled),
        ]:
            button.click(
                edit,
                inputs=inputs
                + [gr.State(f"upload:{kind}"), component]
                + list(settings.values()),
                outputs=outputs,
                **queue,
            )
        selected.input(
            region_images,
            inputs=[state, selected],
            outputs=[mask, texture, geometry, tiled],
            **queue,
        )
    return demo.queue(max_size=16)


if __name__ == "__main__":
    create_demo().launch(css=CSS)
