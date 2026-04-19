# ComfyUI-DynaTrain

ComfyUI-DynaTrain is a custom node pack for building safer LoRA training workflows inside ComfyUI.

The project focuses on three things:

- dataset validation before training
- precision and VRAM planning
- LoRA-oriented training node ergonomics that are easier to wire into ComfyUI graphs

The current implementation defaults to a backendless spec-only flow, while still supporting optional external trainer launches through environment-based backend wiring.

## What Is Included

- `Dataset Sanity Check`
  Detects empty datasets, one-image datasets, and duplicate-heavy datasets.
- `Caption Pair Validator`
  Detects missing or empty captions and warns on repeated captions.
- `Precision Planner`
  Produces training/save precision guidance, optimizer defaults, VRAM safety warnings, and model-family presets.
- `Sample Preview During Training`
  Builds preview prompt/sample settings and can carry connected ComfyUI `model`, `positive`, and `latents` inputs.
- `Train LoRA`
  Workflow-oriented LoRA training spec node with direct `lora`, `loss_map`, and `steps` outputs.
- `Train LoRA Advanced`
  Advanced training-spec node with direct `lora`, `preview_image`, `loss_map`, and `steps` outputs.
- `Plot Loss Graph`
  Turns a `loss_map` payload into a rendered loss chart image and saves a PNG copy under the runtime folder.
- `Training Job Status`
  Reads job metadata, log tail, preview information, and latest loss.
- `Save LoRA Weights`
  Copies generated or selected LoRA weights into the ComfyUI `loras` directory.

## Supported Model Families

- `sd15`
- `sdxl`
- `flux`
- `future_large`

## Supported Precision Options

- `nvfp4`
- `fp8`
- `fp16`
- `bf16`
- `fp32`

`nvfp4` and `fp8` are intentionally treated as guarded or experimental paths and may require backend- and hardware-specific support.

## Key Behaviors

- Blocks training when the dataset contains no images or only one image.
- Blocks training when captions are missing or empty.
- Warns for duplicate-heavy datasets and repeated captions.
- Exposes `resolution_x` and `resolution_y` directly in the training nodes.
- Exposes `existing_lora` from the ComfyUI `loras` folder when available.
- Returns `lora` payloads that can be chained into `Save LoRA Weights`.
- Exposes a visual `preview_image` socket for preview-oriented workflows.

## Current Architecture Notes

- The public ComfyUI nodes are simplified on purpose and hide backend path details.
- By default, `Train LoRA` and `Train LoRA Advanced` stay in spec-only mode even if `execute=true`.
- Optional backend execution can be enabled with environment variables such as `DYNATRAIN_BACKEND_MODE`, `DYNATRAIN_BACKEND_MODULE`, `DYNATRAIN_BACKEND_SCRIPT`, and `DYNATRAIN_BACKEND_PYTHON`.
- The optional launcher expects a backend that accepts `--config <path>`.

## Project Layout

```text
ComfyUI-DynaTrain/
├── __init__.py
├── nodes.py
├── trainers/
│   ├── background_job_runner.py
│   ├── lora_train_advanced.py
│   ├── precision_planner.py
│   └── preview_runner.py
├── validators/
│   ├── caption_pairs.py
│   ├── dataset_sanity.py
│   └── duplicate_checker.py
├── utils/
│   ├── comfy_paths.py
│   ├── config_presets.py
│   ├── lora_io.py
│   ├── runtime.py
│   ├── vram_profiler.py
│   └── warnings.py
├── docs/
│   └── presets.md
└── tests/
    ├── fake_onetrainer_backend.py
    └── test_dynatrain.py
```

## Installation

Clone or copy this folder into your ComfyUI custom nodes directory:

```bash
ComfyUI/custom_nodes/ComfyUI-DynaTrain
```

Then restart ComfyUI.

If you are developing this node pack outside an existing ComfyUI Python environment:

```bash
pip install -r requirements.txt
```

## Optional Backend Wiring

The training nodes now work without a hardcoded backend. Out of the box they generate config/runtime payloads, previews, `lora` payloads, and `loss_map` outputs without requiring OneTrainer or any other trainer package.

If you want `execute=true` to launch a real backend, set one of these environment configurations before starting ComfyUI:

```bash
export DYNATRAIN_BACKEND_MODE=module
export DYNATRAIN_BACKEND_MODULE=your_backend_module
export DYNATRAIN_BACKEND_PYTHON=/path/to/python
```

or

```bash
export DYNATRAIN_BACKEND_MODE=script
export DYNATRAIN_BACKEND_SCRIPT=/path/to/train_backend.py
export DYNATRAIN_BACKEND_PYTHON=/path/to/python
```

Optional:

```bash
export DYNATRAIN_BACKEND_WORKDIR=/path/to/backend/workdir
```

If no backend is configured, `backend_ready` stays `false` and execution remains in safe spec-only mode.

## Example Workflow Shape

1. Run `Dataset Sanity Check`.
2. Run `Caption Pair Validator`.
3. Run `Precision Planner`.
4. Build a preview config with `Sample Preview During Training`.
5. Send those outputs into `Train LoRA Advanced`.
6. Send `lora` into `Save LoRA Weights`.
7. Optionally send `loss_map` into `Plot Loss Graph`.
8. Optionally send `lora` or `train_job` into `Training Job Status`.

## Development

Validation used in this repo:

```bash
python -m compileall nodes.py trainers utils tests
python -m unittest discover -s tests -v
```

## License

Apache-2.0. See [LICENSE](LICENSE).

## Status

This project is in active iteration. The ComfyUI node surface is already usable for workflow construction, validation, planning, LoRA payload chaining, and backendless spec generation, while optional backend execution details are still evolving.
