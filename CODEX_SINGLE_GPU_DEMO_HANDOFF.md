# Finish testing and fixing the single-GPU Quick Trial demo

## Assignment and authorization

The user asks: "continue what you were doing. test and fix the single-GPU demo" and wants unattended Codex execution until testing is finished. Diagnose and fix the existing Quick Trial UI, launch it with working server-side Gemini planning, and validate the real single-A100 inference path through mesh export. Do the work, not just a plan. Routine reversible changes, dependency repairs, process management for the Quick Trial server, live Gemini test requests, and real GPU tests are authorized. Do not repeatedly ask for routine approval. Stop with an evidence-based blocker if user credentials, provider billing/quota, unavailable hardware, or another external dependency truly prevents progress. Never invent successful test results.

Read applicable AGENTS.md and the existing task/docs first. The current request authorizes fixing/testing the already implemented public demo despite the older profiling task's instruction not to begin implementing one. Preserve the original baseline-first discipline before changing inference/profiling code; inspect existing baseline evidence rather than automatically rerunning the entire old profiling campaign. Do not expand this task into architecture work or the entire historical profiling mission.

## Workspace and constraints

- Repository: `/workspace/Text2TactileGraphics-Code`.
- Allowed fork: `RuihanGao/Text2TactileGraphics-Code`; branch: `profile/runpod-a100` (verified in this session).
- Never modify/push upstream. Never push unless explicitly requested. Preserve all existing user changes. Inspect Git status/diff before edits.
- Preserve the existing advanced Gradio application and its running service. Do not kill/restart it merely to take its port. Scope process termination to positively identified Quick Trial/test processes that must be replaced.
- Target Linux x86_64, Python 3.12, one NVIDIA A100 80GB. Verify actual hardware, memory, and active jobs before GPU runs.
- Keep checkpoint/cache settings inherited from the environment. Expected paths:
  - `HF_HOME=/workspace/model_cache/huggingface`
  - `TEXT2TACTILEGRAPHICS_CKPT_DIR=/workspace/model_cache/text2tactilegraphics/ckpt`
  - `DIFFSYNTH_MODEL_BASE_PATH=/workspace/model_cache/diffsynth`
- Use `TEXT2TACTILEGRAPHICS_MODEL_LIFECYCLE=single_a100_80gb` for this demo. Current backend reportedly rejects `dual_a100_80gb`; do not use it.
- No quantization, checkpoint substitutions, retraining, reduced quality, or model architecture changes to force a pass. Use released handlers rather than duplicating inference.
- No subagents unless the user or applicable project instructions explicitly authorize them.
- Never print or commit secret values, credential files, process environments, authentication files, HTTP authorization headers, or raw provider exceptions that may contain keys. Do not enable shell tracing. Report only credential presence and carefully sanitized diagnostics.
- Keep raw profiling artifacts under ignored `profiling/results/`; do not commit large generated models/meshes/logs.

## Existing work and documentation

Read:

- `CODEX_QUICK_TRIAL_TASK.md`
- `QUICK_TRIAL.md`
- `QUICK_TRIAL_VALIDATION.md`
- `README.md`, `pyproject.toml`, `uv.lock`
- `src/text2tactilegraphics/ui/public_app.py`
- `src/text2tactilegraphics/ui/prompt_planner.py`
- `tests/ui/test_quick_trial.py` and relevant pipeline/backend source
- Existing profiling/baseline evidence and single-GPU integration instructions, where relevant.

A previous completion message reported a separate mobile Quick Trial UI with structured planning, progressive feedback, selective reruns, 264 passing tests, phone browser checks, and two real GPU mesh exports using supplied plans. These claims were supplied as history and have NOT been independently revalidated during the present troubleshooting session. Supplied plans bypass live Gemini planning, so those reports do not establish that the complete live flow works.

Known historical limitation: exported meshes reportedly have a watertightness defect. Inspect and report it accurately; do not claim watertightness without checking it. Fix routine export failures in scope, but do not silently redesign algorithms.

## Confirmed current findings

1. `uv` was not available in the user's shell. `/root/.local/bin/uv`, `/usr/local/bin/uv`, and `.venv/bin/uv` were absent when checked. This does not mean the environment is missing.
2. `.venv/bin/python` exists and points to `/root/.local/share/uv/python/cpython-3.12-linux-x86_64-gnu/bin/python3.12`. It executed successfully and located `text2tactilegraphics.ui.public_app`. Python reported 3.12.14. This only verified interpreter/module availability, not all dependencies or full inference.
3. Launching with `.venv/bin/python -m text2tactilegraphics.ui.public_app` uses the existing project environment. Avoid rebuilding it unnecessarily. Restore uv only if useful, preserving the lockfile and installed CUDA compatibility.
4. User originally saw: `Could not finish: Understanding prompt. The text planner needs a server-side GEMINI_API_KEY. Completed stages are saved.`
5. Initially `/workspace/private/gemini.env` existed but loaded NO nonempty `GEMINI_API_KEY`; a Quick Trial process PID 151258 also lacked the key. File permissions were initially 0666. We successfully ran `chmod 700 /workspace/private` and `chmod 600 /workspace/private/gemini.env`.
6. User then privately saved a key using `SAVE_GEMINI_KEY.txt` and said `key saved`. A subsequent subprocess sourced the file and confirmed `Gemini key present: True`. Never display the file contents. No key has been pasted into the conversation.
7. A real call through `GeminiPromptPlanner().plan("a dolphin with wings")` then failed with its generic sanitized `PlannerError`. A direct diagnostic request using the same model, instruction, and Pydantic schema returned `ClientError`, HTTP **400**, status **INVALID_ARGUMENT**. The diagnostic did not classify it as invalid key, quota, billing, leaked key, or unsupported location based on a few string checks; those checks do NOT establish the cause. Do not assume the key is invalid. A structured-output schema incompatibility is one hypothesis to investigate.
8. Attempts to retrieve a safely redacted provider message were interrupted. One later attempt returned execution session ID 81782 but its result was NOT collected. This and other interrupted commands may have partially executed. Recheck processes; do not rely on session IDs being resumable from a new Codex run.
9. Last confirmed port inspection showed Python PID **83181** serving **7860**. It was the ADVANCED application:
   `/workspace/Text2TactileGraphics-Code/.venv/bin/python -u /workspace/Text2TactileGraphics-Code/src/text2tactilegraphics/ui/app.py`
   It had no Gemini key. `http://127.0.0.1:7860/config` title was `Text-based Tactile Graphics Generation` and textboxes included Generation prompt, Style prefix/suffix, Selection prompt and Texture generation prompt. Do not mistake it for Quick Trial or replace it without need.
10. The earlier Quick Trial PID 151258 was no longer found in a later process scan. Identify current state afresh. Other listening ports included nginx on 7861, 8081, 8001, 7270, 3001 and 9091. Check listeners and proxy configuration before choosing a Quick Trial port; do not assume 7861 is free.

## Planner implementation observed

`GeminiPromptPlanner` reads `GEMINI_API_KEY` or legacy `GENAI_API_KEY` from its process environment on each plan call. It does not automatically load `/workspace/private/gemini.env` and does not currently check `GOOGLE_API_KEY`.

Default model is `gemini-2.5-flash`, override `TEXT2TACTILEGRAPHICS_PLANNER_MODEL`. It uses `google.genai.Client`, 30-second HTTP timeout, temperature 0, JSON response MIME type, `response_schema=PromptPlan`, and validates `response.text` with Pydantic. Its broad exception handler masks provider errors from the UI (appropriate for secret safety but inadequate for diagnosis).

The Pydantic schema forbids extra properties, has bounded text fields, a maximum of eight regions, and a patterned ASCII Braille label. Inspect the actual source and installed SDK. Diagnose the HTTP 400 using safe provider detail extraction. Redact the exact key and recognizable key patterns before emitting any provider message, and never dump complete requests/client objects. Prefer explicit safe error code/reason fields. If a schema restriction is rejected, adjust only the provider-facing schema as needed while retaining local validation; add a meaningful regression test. Consult current official Google documentation for supported structured-output features if needed.

## Remaining work, in order

1. Inspect Git state, task/source files, installed dependencies, existing validation evidence, GPU availability, current servers and active test processes. Record branch/SHA, relevant Python/PyTorch/CUDA/GPU versions and cache settings without secrets. Avoid disrupting advanced UI or active user work.
2. Source `/workspace/private/gemini.env` privately in the process that performs live planning; explicitly export `GEMINI_API_KEY`. Check only that it is nonempty. Key saving does not alter already running server environments.
3. Resolve Gemini HTTP 400 with evidence. Test the same schema/model path that Quick Trial uses, not only a trivial unstructured request. If the provider requires an external action such as a replacement credential or billing, document the exact sanitized blocker and necessary user action.
4. Test live plans for the dolphin-with-wings and textured lamp examples from the original task/docs. Validate separation of shape, segmentation regions, texture prompts, and Braille labels. Make minimal fixes and relevant tests if necessary.
5. Provide a reliable project launcher that loads the server-side key privately, uses the existing project Python (or restored uv), preserves persistent caches, sets the single-A100 lifecycle, and explicitly selects the correct host/port. Fail clearly on an empty key. Do not store the secret in project files. Update launch docs to avoid repeating the user's missing-key and wrong-server problems.
6. Start the actual Quick Trial server on a verified appropriate port, preserving advanced UI. Ensure it inherits the key and the intended lifecycle. Verify its HTTP config/title/components; if a browser tool is available, test through the visible UI. Confirm the correct RunPod proxy/access URL from actual configuration; do not fabricate an externally accessible URL.
7. Execute a representative real single-GPU full flow using live planning: input description -> plan -> base image -> requested regions/textures/geometry/tiling -> mesh/Braille/export. Prefer existing default seed 42, 4-step generation, full quality and existing handlers. Use the existing tests/harness where appropriate; do not silently substitute supplied plans for live validation. Avoid running concurrent GPU-heavy jobs that contend with the active service.
8. Validate the second requested example and selective rerun/resume behavior as appropriate to the original task and existing evidence. Check UI progressive feedback, intermediate artifacts, and a real downloadable mesh. Run focused regression tests for fixes and required existing checks. Do not add tests that merely mirror trivial edits.
9. Inspect mesh export validity and report watertightness accurately. Save timestamps, sanitized logs, test results, output locations, and any limits in `QUICK_TRIAL_VALIDATION.md` or a linked new validation report. Distinguish new measurements from inherited claims and browser tests from direct handler tests.
10. Leave a usable Quick Trial service running if possible. Final report must identify changes, exact launch command, verified address/port, live planner and GPU validation results, evidence/artifact paths, and remaining blockers. No pushes.

## Completion standard

Do not declare success just because the key is present, a server answers HTTP, a mocked test passes, or an old supplied-plan GPU test passed. Success requires the actual Quick Trial server to see the saved key, real structured planning to succeed, and the live single-GPU pipeline to produce a downloadable mesh with evidence. If a genuine external blocker makes this impossible, complete independent work and document the blocker precisely rather than looping indefinitely or claiming success.

## Files created during this troubleshooting session

- `GEMINI_KEY_SETUP.txt`: initial private key-saving instructions; original launch uses unavailable uv.
- `QUICK_TRIAL_LAUNCH.txt`: direct project-Python launch instructions.
- `GEMINI_UI_RESTART.txt`: source/check/export/restart instructions.
- `SAVE_GEMINI_KEY.txt`: private interactive key entry using `/dev/tty`, refuses empty/whitespace input, writes outside repo with restricted permissions.

These contain instructions only, not the key. Inspect actual files before modifying. Prior instructions alone did not fix the complete live flow.

## Execution environment notes

Interactive agent shell commands failed under workspace sandboxing with `bwrap: No permissions to create new namespace`. Escalated commands worked but approval waits repeatedly interrupted progress. The user explicitly requests running `codex exec` without approvals to finish testing. The installed CLI supports `--dangerously-bypass-approvals-and-sandbox`, stdin prompt via `-`, `-C`, and `--output-last-message`. This unrestricted mode relies on the outer RunPod/container isolation; continue honoring all task and secret-handling constraints. Do not inspect Codex authentication files. No fixed token budget or additional agent delegation was requested.
