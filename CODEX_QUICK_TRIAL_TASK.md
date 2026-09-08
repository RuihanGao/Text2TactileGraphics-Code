Read the existing advanced Gradio application and reuse its actual generation handlers/core functions.

## Mission

Create a mobile-friendly **Quick Trial** interface for Text2TactileGraphics.

The target user is a conference attendee standing at a poster and trying the system from a phone.

The default workflow must require only:

1. one free-form text prompt;
2. one Generate button.

Do not expose the existing multi-stage research controls by default.

Keep the existing advanced application intact.

Create the quick interface separately, preferably:

`src/text2tactilegraphics/ui/public_app.py`

## Prompt planning

Convert the user's free-form description into a structured plan:

```python
PromptPlan:
    shape_prompt: str
    regions: list[TextureRegion]
    braille_label: str

TextureRegion:
    part_prompt: str
    texture_prompt: str
```

Support multiple texture regions.

Example:

Input:

`a dolphin with wings with an avocado skin texture.`

Expected interpretation:

* shape prompt: `a dolphin with wings`
* part prompt: `dolphin`
* texture prompt: `avocado skin texture`
* Braille: `dolphin`

Example:

`lamp, the base of the lamp has a tree bark texture, and the lamp shade has a cloth_bag texture.`

Expected interpretation:

* shape: lamp
* region: base of the lamp → tree bark texture
* region: lamp shade → cloth_bag texture
* Braille: lamp

Use a lightweight structured-output text planner rather than loading a large image-generation model solely for prompt parsing.

Keep the planner behind an abstraction so its provider can be changed later.

Validate its output with a schema.

## Quick UI

Default mobile-oriented screen:

* title / short instruction;
* large text prompt box;
* a few tappable example prompts;
* large Generate button;
* stage-progress indicator;
* final 3D mesh viewer;
* download button.

Do not require intermediate confirmation.

The system should automatically proceed:

prompt parsing
→ base image
→ automatic region segmentation
→ tactile texture generation for each region
→ tiling
→ final geometry
→ Braille
→ final mesh.

## Progressive feedback

Do not leave the user staring at one spinner.

As stages complete, show concise progress such as:

* Understanding prompt
* Generating object
* Finding texture regions
* Creating tactile texture
* Building tactile graphic
* Done

Where practical, reveal intermediate visual results progressively.

## Advanced accordion

Below the final/default workflow provide a collapsed control:

`Customize intermediate steps`

When expanded, expose:

* parsed shape prompt;
* each region's segmentation prompt;
* each texture prompt;
* Braille label;
* base image;
* segmentation masks;
* texture outputs;
* tiled outputs;
* relevant existing advanced controls.

Users must be able to edit an intermediate result/setting and rerun **from that stage forward**, rather than restarting the entire pipeline unnecessarily.

Reuse existing advanced UI handlers wherever possible.

## Pipeline state

Introduce a coherent state object so Quick and Advanced modes operate on the same artifacts.

Track at least:

* original user prompt;
* parsed PromptPlan;
* base image;
* segmentation masks;
* per-region texture;
* per-region tiled texture;
* Braille label;
* final mesh.

When an upstream stage is changed, invalidate only dependent downstream stages.

## Lifecycle independence

The Quick UI must not hardcode one GPU lifecycle.

It must work when launched with:

`TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb`

and later with:

`TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=dual_a100_80gb`

without frontend changes.

## Mobile design

Optimize for phone use:

* single-column layout;
* large tap targets;
* minimal text;
* collapsed advanced controls;
* no tiny parameter grids in the default view.

## Existing UI

Do not remove or simplify:

`src/text2tactilegraphics/ui/app.py`

It remains the full research/customization UI.

## Validation

Test at least:

`a dolphin with wings with an avocado skin texture.`

and

`lamp, the base of the lamp has a tree bark texture, and the lamp shade has a cloth_bag texture.`

Verify the planner produces sensible structured plans and the full quick workflow completes.

Do not spend this task changing model quality, checkpoints, or GPU lifecycle implementation.

Do not push commits.
