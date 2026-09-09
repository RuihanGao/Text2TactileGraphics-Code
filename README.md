# Text2TactileGraphics

**Text-based Tactile Graphics Generation for the Visually Impaired**  
[Ruihan Gao\*](https://ruihangao.github.io/), [Joonghyuk Shin\*](https://joonghyuk.com/), [Ava Pun](https://avapun.com/), [Jaesik Park](https://jaesik.info/), [Wenzhen Yuan](https://siebelschool.illinois.edu/about/people/all-faculty/yuanwz), and [Jun-Yan Zhu](https://www.cs.cmu.edu/~junyanz/)  
Carnegie Mellon University · Seoul National University · University of Illinois Urbana-Champaign  

[![arXiv](https://img.shields.io/badge/arXiv-2607.22674-b31b1b.svg)](https://arxiv.org/abs/2607.22674)
[![Project Page](https://img.shields.io/badge/Project_Page-Website-blue)](https://ruihangao.github.io/Text2TactileGraphics/)
[![Checkpoints](https://img.shields.io/badge/Hugging_Face-Checkpoints-yellow)](https://huggingface.co/alex4727/text2tactilegraphics_ckpt)
[![Dataset](https://img.shields.io/badge/Hugging_Face-Dataset-yellow)](https://huggingface.co/datasets/alex4727/text2tactilegraphics_data)

## ECCV Quick Trial Demo

A mobile-friendly one-prompt interface accepts a natural-language description,
automatically parses object shape, textured regions, tactile textures and Braille
label, and returns the final 3D mesh. The collapsed Advanced/Customize workflow
supports intermediate editing; the original research interface remains available.

The validated fast path uses **2× NVIDIA A100-SXM4-80GB**, lifecycle
`dual_a100_80gb`, and true `80gb` numerical behavior. GPU 0 keeps base-edit Qwen,
SAM3 and MoGe resident; GPU 1 keeps the shared texture/tiling Qwen bundle resident.
The warm-start service runs on port **8080**, with one active workflow and at most
two pending requests (default projected-wait budget: 180 seconds).

Requirements: Linux x86_64, Python **3.12** (`>=3.12,<3.13`), `uv sync --frozen`,
released checkpoints and persistent model caches (about 100 GB in validation),
a server-side Gemini planner credential and `HF_TOKEN` for model access/gated
weights. The lockfile pins project 0.1.0 and PyTorch **2.11.0+cu128**; validation
used Python 3.12.14 and CUDA runtime 12.8. The research stack below is historical;
use the frozen lockfile for this demo.

Configure `HF_HOME`, `TEXT2TACTILEGRAPHICS_CKPT_DIR`,
`DIFFSYNTH_MODEL_BASE_PATH`, `HF_TOKEN`, and `QUICK_TRIAL_KEY_FILE`.
The latter points to a trusted private shell file outside the repository defining
`GEMINI_API_KEY`; the launcher requires this file even with an inherited key.
It sets `TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE` itself. Secret values stay
server-side and must never appear in frontend code or Git.

```bash
./launch_eccv.sh --check
./launch_eccv.sh
```

Expected bind: **0.0.0.0:8080**. Expose internal HTTP 8080 in RunPod. Warmup may
take several minutes: wait for `/readyz` to return HTTP 200 and `ready: true`,
not a fixed sleep. `/healthz` reports readiness and queue state.
For an intentional shutdown, run:

```bash
.venv/bin/python scripts/stop_eccv.py
```

The helper verifies process command and checkout identity before stopping it.
After shutdown, `./launch_eccv.sh --single --check` and
`./launch_eccv.sh --single` select the slower `single_a100_80gb` fallback on GPU 0.

Known limits: raw exported meshes may be non-watertight; multi-region segmentation
can be imperfect; RunPod URLs are Pod-specific unless fronted by a stable redirect;
GPU concurrency intentionally remains one workflow at a time.

Start with [ECCV_DEMO.md](ECCV_DEMO.md) for deployment and lightweight UI preview,
and [UI_STYLING_GUIDE.md](UI_STYLING_GUIDE.md) for exact safe styling sections.

## System requirements

This project has been tested on Linux x86_64 with the following stack:

- 8× NVIDIA A100-SXM4-80GB, driver 580.x (CUDA 13)
- Python 3.12.13
- PyTorch 2.9.1 + CUDA 12.8 wheels

## Installation

This repo uses the Python project manager [uv](https://docs.astral.sh/uv/).

1. [Install uv](https://docs.astral.sh/uv/getting-started/installation/).
2. Run `uv sync` to create a Python virtual environment with all dependencies installed.
3. Download Text2TactileGraphics checkpoints from Hugging Face Hub:
    ```bash
    uv run --frozen hf download alex4727/text2tactilegraphics_ckpt \
      --repo-type model \
      --local-dir ckpt
    ```
   Place the `ckpt` folder in `~/.cache/text2tactilegraphics/`, or specify its location via the environment variable
   `TEXT2TACTILEGRAPHICS_CKPT_DIR`. If you keep it in the project root, point the runtime config at that directory:
    ```bash
    export TEXT2TACTILEGRAPHICS_CKPT_DIR="$PWD/ckpt"
    ```

## Usage

### Gradio demo

Run the end-to-end Gradio demo with:

```bash
uv run gradio src/text2tactilegraphics/ui/app.py
```

### Mobile Quick Trial

A separate one-prompt interface is available for phone users. See
[Quick Trial setup and customization](QUICK_TRIAL.md) for launch instructions,
planner configuration, and resumable intermediate editing. The advanced demo
above remains unchanged.

### Environment variables

Set the following environment variables as needed. If they are missing at app startup, you will be prompted on the
terminal.

| Variable                    | Purpose                                                     | Default if unset                         | When required                   |
|-----------------------------|-------------------------------------------------------------|------------------------------------------|---------------------------------|
| `HF_TOKEN`                  | HuggingFace Hub access (gated weights, higher rate limits)  | None                                     | Always, when downloading models |
| `GEMINI_API_KEY`            | Google Gemini API                                           | None                                     | Quick Trial planning or Nano Banana |
| `TEXT2TACTILEGRAPHICS_CKPT_DIR`       | Override default location for Text2TactileGraphics custom checkpoints | `~/.cache/text2tactilegraphics/ckpt`               | Optional                        |
| `HF_HOME`                   | Override default location for Hugging Face model weights    | `~/.cache/huggingface`                   | Optional                        |
| `DIFFSYNTH_MODEL_BASE_PATH` | Override default location for DiffSynth model weights       | `./models` relative to the current shell | Optional                        |

## Text-to-Texture Model Training

We delegate Qwen-Image LoRA training to [DiffSynth-Studio](https://github.com/modelscope/diffsynth-studio). The training data is released on HuggingFace in the CSV format expected by DiffSynth:

```bash
export TEXT2TACTILEGRAPHICS_TEXTURE_DATA=/path/to/text2tactilegraphics_data

uv run --frozen hf download alex4727/text2tactilegraphics_data \
  --repo-type dataset \
  --local-dir "$TEXT2TACTILEGRAPHICS_TEXTURE_DATA"
```

The downloaded dataset should have this layout:

```text
$TEXT2TACTILEGRAPHICS_TEXTURE_DATA/
  tactile_data.csv
  images/
    nb_000000.png
    nbp_000000.png
    real_000000.jpg
```

Then run training from the DiffSynth-Studio repository. These instructions are checked against DiffSynth-Studio commit `83eece4faf52ab392ca707ad643ab62ca2f58773`:

```bash
accelerate launch examples/qwen_image/model_training/train.py \
  --dataset_base_path "$TEXT2TACTILEGRAPHICS_TEXTURE_DATA" \
  --dataset_metadata_path "$TEXT2TACTILEGRAPHICS_TEXTURE_DATA/tactile_data.csv" \
  --data_file_keys image \
  --max_pixels 1048576 \
  --model_id_with_origin_paths "Qwen/Qwen-Image:transformer/diffusion_pytorch_model*.safetensors,Qwen/Qwen-Image:text_encoder/model*.safetensors,Qwen/Qwen-Image:vae/diffusion_pytorch_model.safetensors" \
  --learning_rate 1e-4 \
  --num_epochs 100 \
  --remove_prefix_in_ckpt "pipe.dit." \
  --output_path /path/to/output/tactile_qwen_lora \
  --lora_base_model "dit" \
  --lora_target_modules "to_q,to_k,to_v,add_q_proj,add_k_proj,add_v_proj,to_out.0,to_add_out,img_mlp.net.2,img_mod.1,txt_mlp.net.2,txt_mod.1" \
  --lora_rank 32 \
  --use_gradient_checkpointing \
  --dataset_num_workers 8 \
  --find_unused_parameters \
  --save_steps 100 \
  --enable_wandb_log \
  --gradient_accumulation_steps 4
```

Configure `accelerate` for your local hardware before launching (e.g., # of gpus/processes). Our released texture LoRA was trained on 8x A100 80GB GPUs with per-gpu batch size of 1 and gradient accumulation 4, giving an effective batch size of 32. We stopped at 3,000 steps after validation; you can stop earlier or later based on your own validation samples. 

## Development

### Project structure

- `src/text2tactilegraphics/`: Main source code.
    - `assets/`: Image assets used during generation, and example assets for the Gradio app.
    - `generation/`: Image generation, texture generation, and segmentation.
    - `geometry/`: Mesh and braille creation.
    - `ui/`: Gradio interface.
- `tests/`: Testing code.

### Code formatting and linting

This project uses [ruff](https://docs.astral.sh/ruff/) for formatting and linting:

```bash
uv run --frozen ruff format src/
uv run --frozen ruff check --fix src/
```

### Testing

This project uses [pytest](https://docs.pytest.org/) for tests. Run core tests with:

```bash
uv run --frozen pytest -q tests
```

#### Slow tests

End-to-end tests that run inference on a CUDA GPU are marked `@pytest.mark/slow` and **skipped by default**. To run
them, use:

```bash
uv run --frozen pytest -q tests -m slow
```

#### Regression tests

Regression tests pin outputs against snapshot files committed under `tests/<package>/<test_module_stem>/`. When a
snapshot *intentionally* changes (e.g. due to an algorithm change), you can update these snapshots with:

```bash
uv run --frozen pytest -q tests --force-regen
```

After regenerating, commit the updated snapshot files alongside the code change.

#### Debugging test outputs

Some tests save additional outputs to `/tmp/pytest-of-<username>/pytest-<number>/` to assist with visual debugging. You
can change this output directory with

```bash
uv run --frozen pytest -q tests --basetemp <output_directory>
```

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{gao2026text2tactilegraphics,
  title     = {Text-based Tactile Graphics Generation for the Visually Impaired},
  author    = {Gao, Ruihan and Shin, Joonghyuk and Pun, Ava and Park, Jaesik and Yuan, Wenzhen and Zhu, Jun-Yan},
  booktitle = {European Conference on Computer Vision (ECCV)},
  year      = {2026}
}
```

## Acknowledgments

This codebase is released with a clean Git history. All students (Ruihan Gao, Joonghyuk Shin, and Ava Pun) made substantial contributions to both the research project and code development.
<!-- Add other Acks here -->

### Single A100 80GB deployment

Use the validated sequential Qwen lifecycle with the existing advanced UI:

```bash
TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb \
GRADIO_SERVER_NAME=0.0.0.0 GRADIO_SERVER_PORT=7860 \
uv run --frozen gradio src/text2tactilegraphics/ui/app.py
```

This explicit policy requires true `80gb` numerical mode and all model roles on
GPU 0. It releases the base, texture, and tiling Qwen bundles after each call,
including generator-owned LoRA references, while retaining SAM3/MoGe. It uses the
existing unload API and checks reclamation; it does not force CPU offload or
change generation presets. GPU handlers share one Gradio concurrency group.
The debug GPU/VRAM controls are locked in this policy; other advanced controls
remain available. Lifecycle memory readings use the generation model logger at
DEBUG level. Set `TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=cached` (the default) to
retain the research application's original caching/configuration behavior.

This policy uses separate Qwen loads for texture and tiling. The approximately
75-second profiling result additionally reused their compatible bundle; that
optimization is not part of this explicit unload-after-each-stage policy.
See `profiling/results/APPLICATION_INTEGRATION_REPORT.md` for measured application
latency, memory, output equivalence, and remaining demo limitations.
