Fixed Gemini’s schema error and unavailable planner model, and added a launcher that loads the saved key automatically.

Verified:
- Both examples completed live Gemini → single-A100 inference → browser mesh download.
- Braille-only rerun passed in 47 seconds, preserving intermediate images.
- 264 core tests passed.

Quick Trial remains running on **8080**, accessible locally through nginx on **8081**. The advanced app remains on **7860**.

Relaunch command: `./launch_quick_trial.sh`

Remaining limits: meshes are **not watertight**, external RunPod access is unverified, and the workspace volume does not enforce credential-file permission changes.

[Validation report and artifact locations](/workspace/Text2TactileGraphics-Code/QUICK_TRIAL_LIVE_VALIDATION.md). No pushes.