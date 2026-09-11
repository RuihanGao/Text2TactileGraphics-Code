"""Mobile Quick Trial; run with python -m text2tactilegraphics.ui.public_app."""

import os
import uuid
from contextlib import nullcontext
from html import escape

import gradio as gr
from pydantic import ValidationError

from text2tactilegraphics.generation.utils import mask_to_image
from text2tactilegraphics.ui.admission import Admission, request_owner
from text2tactilegraphics.ui.input_log import record_input
from text2tactilegraphics.ui.prompt_planner import (
    GeminiPromptPlanner,
    InvalidPromptError,
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


def create_demo(
    planner=None, pipeline=None, service=None, admission=None, moderator=None
):
    moderator = moderator or GeminiPromptPlanner()
    pipeline = pipeline or QuickPipeline(planner or GeminiPromptPlanner())
    if admission is None and os.getenv("TEXT2TACTILEGRAPHICS_POSTER_MODE") == "1":
        admission = Admission.from_environment()
    with gr.Blocks(
        title="Text2TactileGraphics · Quick Trial", delete_cache=(3600, 86400)
    ) as demo:
        state = gr.State(QuickState())
        ticket = gr.Textbox(value="", visible=False)
        gr.Markdown(
            "# Quick Trial\nDescribe an object and its textures. We’ll turn it into a tactile graphic."
        )
        prompt = gr.Textbox(
            label="What would you like to create?",
            placeholder="A dolphin with wings and avocado skin texture…",
            lines=4,
            max_length=2000,
        )
        gr.Markdown(
            "Submitted prompts, customization fields and settings are logged to "
            "understand user preferences. Please do not include personal information."
        )
        gr.Examples(
            [[text] for text in EXAMPLES], inputs=prompt, label="Try an example"
        )
        if service is not None or admission is not None:
            service_status = gr.HTML(status_html("Service: Warming"))
            service_timer = gr.Timer(2)
        generate = gr.Button(
            "Generate",
            variant="primary",
            size="lg",
            interactive=service is None or service.state == "Ready",
        )
        if service is not None or admission is not None:

            def readiness(request: gr.Request):
                ready = service is None or service.state == "Ready"
                owner = request_owner(request)
                queue_state = admission.snapshot(owner) if admission else None
                label = service.state if service else "Ready"
                if (
                    ready
                    and queue_state
                    and (
                        queue_state["active"]
                        or queue_state["pending"]
                        or queue_state["scheduled"]
                    )
                ):
                    label = "Busy"
                    ahead = queue_state["jobs_ahead"]
                    if ahead is not None:
                        label += f" · {ahead} jobs ahead"
                        if ahead:
                            label += f" · about {queue_state['wait_bucket_minutes']} minute(s)"
                notice = admission.notice(owner) if admission else ""
                return status_html(f"Service: {label}. {notice}"), gr.Button(
                    interactive=ready and not (admission and admission.owns(owner))
                )

            service_timer.tick(
                readiness,
                outputs=[service_status, generate],
                queue=False,
                api_visibility="private",
            )
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
                generate: gr.Button(
                    interactive=s.status == "Done"
                    or s.status.startswith("Could not finish:")
                ),
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
            generate,
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
                if isinstance(exc, InvalidPromptError):
                    s.status = str(exc)
                    yield snapshot(s, index)
                    return
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

        def start(text, token="", request: gr.Request = None):
            # A new Generate request starts an isolated session artifact graph.
            s = QuickState()

            def work():
                if service is not None:
                    service.require_ready()
                with (
                    admission.run(token, request_owner(request))
                    if admission
                    else nullcontext()
                ):
                    yield from pipeline.generate(s, text)

            yield from stream(s, 0, work())

        def edit(
            token,
            request: gr.Request,
            s,
            index,
            shape_text,
            rows,
            label,
            action,
            image,
            *values,
        ):
            def work():
                if service is not None:
                    service.require_ready()
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
                if plan != s.plan:
                    moderator.validate_prompt(plan.model_dump_json())
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

            def admitted_work():
                with (
                    admission.run(token, request_owner(request))
                    if admission
                    else nullcontext()
                ):
                    yield from work()

            yield from stream(s, index, admitted_work())

        queue = {
            "concurrency_id": "quick_trial",
            "concurrency_limit": 1,
            "api_visibility": "private",
            "trigger_mode": "once",
        }

        def reserve(request: gr.Request):
            try:
                if service is not None:
                    service.require_ready()
                token = admission.reserve(request_owner(request))
                return token, gr.Button(interactive=False)
            except ValueError as exc:
                raise gr.Error(str(exc), title="Live demo busy") from None

        def bind(button, fn, inputs):
            def record_submission(*values):
                if fn is start:
                    record_input("generate", {"prompt": values[0]})
                else:
                    record_input(
                        "customize",
                        {
                            "region_index": values[2],
                            "shape_prompt": values[3],
                            "regions": values[4],
                            "braille_label": values[5],
                            "action": values[6],
                            "image": values[7],
                            "settings": dict(zip(settings, values[8:])),
                        },
                    )

            recorded = button.click(
                record_submission,
                inputs=inputs,
                queue=False,
                api_visibility="private",
                trigger_mode="once",
            )
            if admission is None:
                return recorded.success(fn, inputs=inputs, outputs=outputs, **queue)
            accepted = recorded.success(
                reserve,
                outputs=[ticket, generate],
                queue=False,
                api_visibility="private",
                trigger_mode="once",
            )
            return accepted.success(fn, inputs=inputs, outputs=outputs, **queue)

        bind(generate, start, [prompt, ticket])
        inputs = [ticket, state, selected, shape, regions, braille]
        # Every GPU action uses the same queue across sessions and stages.
        bind(
            apply,
            edit,
            inputs=inputs
            + [gr.State("apply"), gr.State(None)]
            + list(settings.values()),
        )
        bind(
            rerun,
            edit,
            inputs=inputs + [stage, gr.State(None)] + list(settings.values()),
        )
        for button, kind, component in [
            (use_base, "base", base),
            (use_mask, "segmentation", mask),
            (use_texture, "texture", texture),
            (use_geometry, "geometry", geometry),
            (use_tiled, "tiling", tiled),
        ]:
            bind(
                button,
                edit,
                inputs=inputs
                + [gr.State(f"upload:{kind}"), component]
                + list(settings.values()),
            )
        selected.input(
            region_images,
            inputs=[state, selected],
            outputs=[mask, texture, geometry, tiled],
            **queue,
        )
    demo.poster_admission = admission
    return demo.queue(
        max_size=admission.max_pending + 2 if admission else 16, api_open=False
    )


def create_server(service, pipeline=None, admission=None):
    """Mount diagnostics outside the Gradio inference queue."""
    from fastapi import FastAPI
    from fastapi.responses import JSONResponse

    from text2tactilegraphics.config import get_total_gpus, global_config

    demo = create_demo(service=service, pipeline=pipeline, admission=admission)
    app = FastAPI(docs_url=None, redoc_url=None, openapi_url=None)

    @app.middleware("http")
    async def browser_identity(request, call_next):
        response = await call_next(request)
        if not request.cookies.get("eccv_client"):
            response.set_cookie(
                "eccv_client",
                uuid.uuid4().hex,
                httponly=True,
                samesite="lax",
                max_age=86400,
            )
        return response

    def health():
        return {
            "process_alive": True,
            "gpu_count": get_total_gpus(),
            "lifecycle": global_config().model_lifecycle,
            "state": service.state,
            "ready": service.state == "Ready",
            "queue": demo.poster_admission.snapshot()
            if demo.poster_admission
            else {"pending": demo._queue.get_status().queue_size},
        }

    @app.get("/healthz")
    def alive():
        return health()

    @app.get("/readyz")
    def ready():
        result = health()
        return JSONResponse(result, status_code=200 if result["ready"] else 503)

    return gr.mount_gradio_app(app, demo, path="/", css=CSS, show_error=False)


if __name__ == "__main__":
    import logging
    import os

    import uvicorn

    from text2tactilegraphics.ui.inference_service import InferenceService

    logging.basicConfig(level=logging.INFO)
    logging.getLogger("text2tactilegraphics.geometry").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)
    service = InferenceService()
    service.start()
    pod_id = os.environ.get("RUNPOD_POD_ID")
    if pod_id:
        logging.info("RunPod proxy candidate: https://%s-8080.proxy.runpod.net", pod_id)
    uvicorn.run(
        create_server(service),
        host=os.getenv("GRADIO_SERVER_NAME", "0.0.0.0"),
        port=int(os.getenv("GRADIO_SERVER_PORT", "8080")),
        access_log=False,
        workers=1,
    )
