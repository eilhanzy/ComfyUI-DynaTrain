# DynaTrain Presets

These presets follow the PDF design goal: train safer, not just faster.

## Model Families

- `sd15`: prefers `512` or `768` resolution with `AdamW8bit` and conservative accumulation.
- `sdxl`: defaults to `1024` resolution and stronger caching because VRAM pressure rises quickly.
- `flux`: uses `1024` resolution with batch size `1` and heavier accumulation because memory pressure is substantially higher.
- `future_large`: keeps batch size at `1` and relies on higher accumulation to stay inside safe memory bounds.

## VRAM Tiers

- `small` (`< 8 GB`): lowest-risk preset, disk caching on, aggressive accumulation.
- `medium` (`8 GB - 15.5 GB`): balanced preset for most consumer GPUs.
- `large` (`>= 16 GB`): safer headroom for larger batch sizes or fewer accumulation steps.

## Precision Defaults

- Auto precision resolves to `FP16` unless BF16 support is explicitly enabled and VRAM is healthy.
- `FP8` stays experimental and intentionally requires an opt-in flag.
- Save precision defaults to the selected training precision, except that unsupported auto paths fall back to `FP16`.
- OneTrainer launch plans receive these precision values in the generated config artifact and pass that config by default as `--config <path>`.

## Safety Rules Reflected In Code

- Training is blocked when only one image is present.
- Training is blocked when caption mapping is broken or captions are empty.
- Duplicate-heavy datasets and repeated captions raise warnings.
- High accumulation values raise a VRAM safety warning.
- Experimental or unstable precision requests raise explicit warnings.

## OneTrainer Runtime

- The internal launcher still supports dry-run and background job execution, but the ComfyUI-facing training nodes now expose a simplified, training-centric surface.
- `Train LoRA` is the more workflow-oriented node with `model`, `latents`, `positive` inputs and direct `lora`, `loss_map`, `steps` outputs.
- `Train LoRA Advanced` now returns `lora`, `preview_image`, `loss_map`, and `steps`, so it can feed `Save LoRA Weights` directly and also expose a visual preview socket in ComfyUI.
- Both training nodes now expose `resolution_x` and `resolution_y` so non-square training sizes can be wired directly from the UI.
- Backend path fields such as `python_executable`, `module_name`, `script_path`, `backend_workdir`, `execute`, and manual output paths are intentionally hidden from the ComfyUI-facing training nodes.
- Exposed dtype controls now cover `nvfp4`, `fp8`, `fp16`, `bf16`, and `fp32` for both training and LoRA weight precision.
- `Save LoRA Weights` accepts the `lora` payload and writes a `.safetensors` file into the `loras` directory.
- `Training Job Status` also exposes a `preview_image` output alongside loss and log details.
- Background jobs persist metadata under `runtime/jobs/<job_id>.json`.
- Combined stdout/stderr logs are written to `runtime/logs/<job_id>.log`.
- Preview assets are expected under `runtime/previews/<job_id>/`.
- The generated OneTrainer config artifact is written under the selected `output_dir`.
- Generated training configs carry both `resolution_x` and `resolution_y`, plus a mirrored `resolution: {x, y}` block for clearer downstream mapping.
- Entrypoints support both `python -m <module_name>` and `python <script_path>` launch styles.
- v1 assumes the backend accepts `--config <generated_config_path>`.
- If `execute=True` and the backend entrypoint is missing, the node now fails early with an install/path guidance message instead of spawning a broken background job.
- Dry-run and status payloads also expose `backend_ready` and `backend_check_message` so backend availability is visible before launch.
- `existing_lora` choices are sourced from the ComfyUI `loras` folder when available, with a local fallback to a repo `loras/` folder outside ComfyUI.
