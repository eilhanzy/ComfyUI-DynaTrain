from __future__ import annotations

import json
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, Tuple

from .trainers.lora_train_advanced import (
    build_lora_payload,
    build_loss_map,
    get_job_status,
    prepare_training_run,
)
from .trainers.precision_planner import plan_precision
from .trainers.preview_runner import build_preview_config
from .utils.comfy_paths import list_lora_files
from .utils.lora_io import save_lora_weights
from .utils.runtime import default_runtime_root, loss_history
from .validators.caption_pairs import validate_caption_pairs
from .validators.dataset_sanity import summarize_dataset


DTYPE_CHOICES = ["nvfp4", "fp8", "fp16", "bf16", "fp32"]


def _to_pretty_json(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True)


def _normalize_optimizer_choice(value: str) -> str:
    mapping = {
        "AdamW": "adamw",
        "AdamW8bit": "adamw8bit",
        "Lion": "lion",
        "PagedAdamW8bit": "paged_adamw8bit",
    }
    return mapping[value]


def _default_output_name(dataset_dir: str, fallback: str = "dynatrain_lora") -> str:
    dataset_name = Path(dataset_dir).expanduser().name.strip() if dataset_dir.strip() else ""
    return dataset_name or fallback


def _default_output_dir() -> str:
    return str(default_runtime_root() / "generated_loras")


def _default_base_model_path(model: Any) -> str:
    return "[comfy_model_input]" if model is not None else "[base_model_unspecified]"


def _default_backend_workdir() -> str:
    return str(Path(__file__).resolve().parent)


def _pil_to_comfy_image(image: Any):
    try:
        import numpy as np
        import torch
    except ImportError:
        return None

    rgb_image = image.convert("RGB")
    array = np.asarray(rgb_image).astype("float32") / 255.0
    return torch.from_numpy(array).unsqueeze(0)


def _render_preview_placeholder(title: str, body: str):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None

    image = Image.new("RGB", (768, 432), color=(26, 28, 33))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((24, 24, 744, 408), radius=18, fill=(36, 39, 46))
    draw.text((48, 48), title, fill=(220, 232, 244))
    wrapped = textwrap.fill(body or "No preview prompt configured.", width=42)
    draw.text((48, 108), wrapped, fill=(196, 202, 214), spacing=6)
    draw.text((48, 344), "Preview image will appear here once training samples exist.", fill=(133, 178, 154))
    return _pil_to_comfy_image(image)


def _resolve_preview_image(latest_preview_path: str, fallback_text: str):
    preview_path = latest_preview_path.strip()
    if preview_path:
        candidate = Path(preview_path)
        if candidate.exists():
            try:
                from PIL import Image
            except ImportError:
                return None
            with Image.open(candidate) as image:
                return _pil_to_comfy_image(image)
    return _render_preview_placeholder("DynaTrain Preview", fallback_text)


def _render_loss_graph(points: list[Dict[str, Any]], title: str):
    try:
        from PIL import Image, ImageDraw
    except ImportError:
        return None, None

    width = 1024
    height = 576
    margin_left = 84
    margin_right = 36
    margin_top = 56
    margin_bottom = 72
    image = Image.new("RGB", (width, height), color=(23, 25, 30))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle((18, 18, width - 18, height - 18), radius=20, fill=(34, 37, 45))
    draw.text((40, 34), title, fill=(224, 232, 241))

    if not points:
        draw.text((40, 120), "No loss samples found in the provided loss_map/log.", fill=(210, 170, 170))
        return image, _pil_to_comfy_image(image)

    graph_left = margin_left
    graph_top = margin_top
    graph_right = width - margin_right
    graph_bottom = height - margin_bottom
    draw.rectangle((graph_left, graph_top, graph_right, graph_bottom), outline=(88, 96, 110), width=1)

    min_loss = min(point["loss"] for point in points)
    max_loss = max(point["loss"] for point in points)
    if max_loss == min_loss:
        max_loss = min_loss + 1.0

    step_min = min(point["step"] for point in points)
    step_max = max(point["step"] for point in points)
    if step_max == step_min:
        step_max = step_min + 1

    for grid_index in range(5):
        y = graph_top + int((graph_bottom - graph_top) * (grid_index / 4))
        draw.line((graph_left, y, graph_right, y), fill=(58, 63, 74), width=1)
        value = max_loss - ((max_loss - min_loss) * (grid_index / 4))
        draw.text((24, y - 8), f"{value:.4f}", fill=(166, 176, 190))

    polyline: list[tuple[int, int]] = []
    for point in points:
        x_ratio = (point["step"] - step_min) / (step_max - step_min)
        y_ratio = (point["loss"] - min_loss) / (max_loss - min_loss)
        x = graph_left + int((graph_right - graph_left) * x_ratio)
        y = graph_bottom - int((graph_bottom - graph_top) * y_ratio)
        polyline.append((x, y))

    if len(polyline) == 1:
        x, y = polyline[0]
        draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(109, 226, 168))
    else:
        draw.line(polyline, fill=(109, 226, 168), width=3)
        for x, y in polyline[-3:]:
            draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=(222, 246, 233))

    draw.text((graph_left, height - 46), f"steps: {step_min} -> {step_max}", fill=(166, 176, 190))
    draw.text((graph_right - 160, height - 46), f"latest loss: {points[-1]['loss']:.4f}", fill=(166, 176, 190))
    return image, _pil_to_comfy_image(image)


def _save_loss_graph_image(image: Any, filename_prefix: str, loss_map: Dict[str, Any]) -> str:
    if image is None:
        return ""

    runtime_root = default_runtime_root()
    target_dir = runtime_root / "loss_graphs"
    target_dir.mkdir(parents=True, exist_ok=True)

    normalized_prefix = filename_prefix.strip() or "loss_graph"
    normalized_prefix = normalized_prefix.lstrip("/").replace("\\", "/")
    if normalized_prefix.startswith("loss_graphs/"):
        normalized_prefix = normalized_prefix[len("loss_graphs/") :]

    relative_path = Path(normalized_prefix)
    if relative_path.suffix.lower() != ".png":
        job_id = loss_map.get("job_id", "").strip()
        suffix = f"_{job_id}" if job_id else ""
        relative_path = Path(f"{relative_path}{suffix}.png")

    full_path = target_dir / relative_path
    full_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(full_path, format="PNG")
    return str(full_path)


class DatasetSanityCheckNode:
    CATEGORY = "DynaTrain/Validation"
    FUNCTION = "run"
    RETURN_TYPES = ("DYNA_DATASET_REPORT", "STRING", "BOOLEAN")
    RETURN_NAMES = ("dataset_report", "summary", "blocked")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_dir": ("STRING", {"default": ""}),
                "caption_extension": ("STRING", {"default": ".txt"}),
                "recursive": ("BOOLEAN", {"default": True}),
            }
        }

    def run(
        self,
        dataset_dir: str,
        caption_extension: str = ".txt",
        recursive: bool = True,
    ) -> Tuple[Dict[str, Any], str, bool]:
        dataset_report = summarize_dataset(
            dataset_dir=dataset_dir,
            caption_extension=caption_extension,
            recursive=recursive,
        )
        return dataset_report, _to_pretty_json(dataset_report), dataset_report["blocked"]


class PrecisionPlannerNode:
    CATEGORY = "DynaTrain/Planning"
    FUNCTION = "run"
    RETURN_TYPES = ("DYNA_PRECISION_PLAN", "STRING", "STRING")
    RETURN_NAMES = ("precision_plan", "summary", "warnings_text")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_family": (["sd15", "sdxl", "flux", "future_large"], {"default": "sd15"}),
                "vram_gb": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 96.0, "step": 0.5}),
                "requested_precision": (
                    ["auto", "nvfp4", "fp8", "fp16", "bf16", "fp32"],
                    {"default": "auto"},
                ),
                "save_precision": (
                    ["auto", "nvfp4", "fp8", "fp16", "bf16", "fp32"],
                    {"default": "auto"},
                ),
                "optimizer": (
                    ["adamw8bit", "adamw", "lion", "paged_adamw8bit"],
                    {"default": "adamw8bit"},
                ),
                "gradient_accumulation_steps": (
                    "INT",
                    {"default": 1, "min": 1, "max": 128, "step": 1},
                ),
                "gpu_supports_bf16": ("BOOLEAN", {"default": True}),
                "allow_experimental_fp8": ("BOOLEAN", {"default": False}),
            }
        }

    def run(
        self,
        model_family: str,
        vram_gb: float,
        requested_precision: str,
        save_precision: str,
        optimizer: str,
        gradient_accumulation_steps: int,
        gpu_supports_bf16: bool,
        allow_experimental_fp8: bool,
    ) -> Tuple[Dict[str, Any], str, str]:
        plan = plan_precision(
            model_family=model_family,
            vram_gb=vram_gb,
            requested_precision=requested_precision,
            save_precision=save_precision,
            optimizer=optimizer,
            gradient_accumulation_steps=gradient_accumulation_steps,
            gpu_supports_bf16=gpu_supports_bf16,
            allow_experimental_fp8=allow_experimental_fp8,
        )
        warning_text = "\n".join(plan["warnings"]) if plan["warnings"] else ""
        return plan, _to_pretty_json(plan), warning_text


class CaptionPairValidatorNode:
    CATEGORY = "DynaTrain/Validation"
    FUNCTION = "run"
    RETURN_TYPES = ("DYNA_VALIDATED_DATASET", "STRING", "BOOLEAN")
    RETURN_NAMES = ("validated_dataset", "summary", "blocked")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "dataset_report": ("DYNA_DATASET_REPORT",),
                "warn_repeated_caption_ratio": (
                    "FLOAT",
                    {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.05},
                ),
            }
        }

    def run(
        self,
        dataset_report: Dict[str, Any],
        warn_repeated_caption_ratio: float = 0.3,
    ) -> Tuple[Dict[str, Any], str, bool]:
        validated_dataset = validate_caption_pairs(
            dataset_report=dataset_report,
            warn_repeated_caption_ratio=warn_repeated_caption_ratio,
        )
        return (
            validated_dataset,
            _to_pretty_json(validated_dataset),
            validated_dataset["blocked"],
        )


class SamplePreviewDuringTrainingNode:
    CATEGORY = "DynaTrain/Training"
    FUNCTION = "run"
    RETURN_TYPES = ("DYNA_PREVIEW_CONFIG", "STRING")
    RETURN_NAMES = ("preview_config", "summary")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "enabled": ("BOOLEAN", {"default": True}),
                "sample_every_n_steps": (
                    "INT",
                    {"default": 200, "min": 0, "max": 100000, "step": 10},
                ),
                "sample_prompts": (
                    "STRING",
                    {
                        "default": "portrait photo, soft cinematic light",
                        "multiline": True,
                    },
                ),
                "sampler": (
                    ["euler_a", "euler", "dpmpp_2m", "dpmpp_sde"],
                    {"default": "euler_a"},
                ),
                "sample_steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5}),
                "seed": ("INT", {"default": 42, "min": 0, "max": 2147483647, "step": 1}),
            },
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latents": ("LATENT",),
            },
        }

    def run(
        self,
        enabled: bool,
        sample_every_n_steps: int,
        sample_prompts: str,
        sampler: str,
        sample_steps: int,
        cfg_scale: float,
        seed: int,
        model: Any = None,
        positive: Any = None,
        latents: Any = None,
    ) -> Tuple[Dict[str, Any], str]:
        preview_config = build_preview_config(
            enabled=enabled,
            sample_every_n_steps=sample_every_n_steps,
            sample_prompts=sample_prompts,
            sampler=sampler,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            model=model,
            positive=positive,
            latents=latents,
        )
        return preview_config, _to_pretty_json(preview_config)


class TrainLoRANode:
    CATEGORY = "DynaTrain/Training"
    FUNCTION = "run"
    RETURN_TYPES = ("DYNA_LORA", "DYNA_LOSS_MAP", "INT")
    RETURN_NAMES = ("lora", "loss_map", "steps")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "latents": ("LATENT",),
                "positive": ("CONDITIONING",),
                "dataset_dir": ("STRING", {"default": ""}),
                "model_family": (["sd15", "sdxl", "flux", "future_large"], {"default": "sd15"}),
                "batch_size": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "grad_accumulation_steps": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1}),
                "steps": ("INT", {"default": 16, "min": 1, "max": 1000000, "step": 1}),
                "learning_rate": ("FLOAT", {"default": 0.0005, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "rank": ("INT", {"default": 8, "min": 1, "max": 512, "step": 1}),
                "resolution_x": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "resolution_y": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "optimizer": (
                    ["AdamW", "AdamW8bit", "Lion", "PagedAdamW8bit"],
                    {"default": "AdamW"},
                ),
                "loss_function": (
                    ["MSE", "Huber", "MAE"],
                    {"default": "MSE"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "control_after_generate": (
                    ["randomize", "fixed", "increment"],
                    {"default": "randomize"},
                ),
                "training_dtype": (DTYPE_CHOICES, {"default": "bf16"}),
                "lora_dtype": (DTYPE_CHOICES, {"default": "bf16"}),
                "quantized_backward": ("BOOLEAN", {"default": False}),
                "algorithm": (
                    ["LoRA", "LoCon", "LoHa"],
                    {"default": "LoRA"},
                ),
                "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                "checkpoint_depth": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "offloading": ("BOOLEAN", {"default": False}),
                "existing_lora": (list_lora_files(), {"default": "[None]"}),
                "bucket_mode": ("BOOLEAN", {"default": False}),
                "bypass_mode": ("BOOLEAN", {"default": False}),
                "sample_every_n_steps": ("INT", {"default": 200, "min": 0, "max": 100000, "step": 10}),
                "sample_prompts": ("STRING", {"default": "portrait photo, soft cinematic light", "multiline": True}),
                "sampler": (
                    ["euler_a", "euler", "dpmpp_2m", "dpmpp_sde"],
                    {"default": "euler_a"},
                ),
                "sample_steps": ("INT", {"default": 20, "min": 1, "max": 150, "step": 1}),
                "cfg_scale": ("FLOAT", {"default": 7.0, "min": 1.0, "max": 30.0, "step": 0.5}),
            }
        }

    def run(
        self,
        model: Any,
        latents: Any,
        positive: Any,
        dataset_dir: str,
        model_family: str,
        batch_size: int,
        grad_accumulation_steps: int,
        steps: int,
        learning_rate: float,
        rank: int,
        resolution_x: int,
        resolution_y: int,
        optimizer: str,
        loss_function: str,
        seed: int,
        control_after_generate: str,
        training_dtype: str,
        lora_dtype: str,
        quantized_backward: bool,
        algorithm: str,
        gradient_checkpointing: bool,
        checkpoint_depth: int,
        offloading: bool,
        existing_lora: str,
        bucket_mode: bool,
        bypass_mode: bool,
        sample_every_n_steps: int,
        sample_prompts: str,
        sampler: str,
        sample_steps: int,
        cfg_scale: float,
    ) -> Tuple[Dict[str, Any], Dict[str, Any], int]:
        dataset_report = summarize_dataset(
            dataset_dir=dataset_dir,
            caption_extension=".txt",
            recursive=True,
        )
        validated_dataset = validate_caption_pairs(dataset_report=dataset_report)
        precision_plan = plan_precision(
            model_family=model_family,
            vram_gb=12.0,
            requested_precision=training_dtype,
            save_precision=lora_dtype,
            optimizer=_normalize_optimizer_choice(optimizer),
            gradient_accumulation_steps=grad_accumulation_steps,
            gpu_supports_bf16=True,
            allow_experimental_fp8=True,
        )
        preview_config = build_preview_config(
            enabled=True,
            sample_every_n_steps=sample_every_n_steps,
            sample_prompts=sample_prompts,
            sampler=sampler,
            sample_steps=sample_steps,
            cfg_scale=cfg_scale,
            seed=seed,
            model=model,
            positive=positive,
            latents=latents,
        )
        train_job = prepare_training_run(
            validated_dataset=validated_dataset,
            precision_plan=precision_plan,
            preview_config=preview_config,
            base_model_path=_default_base_model_path(model),
            output_dir=_default_output_dir(),
            output_name=_default_output_name(dataset_dir),
            backend_workdir=_default_backend_workdir(),
            python_executable=sys.executable,
            entrypoint_mode="module",
            module_name="onetrainer",
            script_path="",
            network_dim=rank,
            network_alpha=rank,
            learning_rate=learning_rate,
            max_train_steps=steps,
            lr_scheduler="constant",
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            batch_size_override=batch_size,
            gradient_accumulation_override=grad_accumulation_steps,
            execute=False,
            extra_args="",
            runtime_root="",
            model=model,
            positive=positive,
            latents=latents,
            optimizer=optimizer,
            training_dtype=training_dtype,
            lora_dtype=lora_dtype,
            loss_function=loss_function,
            seed=seed,
            control_after_generate=control_after_generate,
            quantized_backward=quantized_backward,
            algorithm=algorithm,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_depth=checkpoint_depth,
            offloading=offloading,
            existing_lora=existing_lora,
            bucket_mode=bucket_mode,
            bypass_mode=bypass_mode,
        )
        return build_lora_payload(train_job), build_loss_map(train_job), steps


class TrainLoRAAdvancedNode:
    CATEGORY = "DynaTrain/Training"
    FUNCTION = "run"
    RETURN_TYPES = ("DYNA_LORA", "IMAGE", "DYNA_LOSS_MAP", "INT")
    RETURN_NAMES = ("lora", "preview_image", "loss_map", "steps")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "validated_dataset": ("DYNA_VALIDATED_DATASET",),
                "precision_plan": ("DYNA_PRECISION_PLAN",),
                "preview_config": ("DYNA_PREVIEW_CONFIG",),
                "rank": ("INT", {"default": 16, "min": 1, "max": 512, "step": 1}),
                "learning_rate": ("FLOAT", {"default": 0.0001, "min": 0.0, "max": 1.0, "step": 0.00001}),
                "max_train_steps": ("INT", {"default": 2000, "min": 1, "max": 1000000, "step": 50}),
                "lr_scheduler": (
                    ["constant", "cosine", "cosine_with_restarts", "linear"],
                    {"default": "cosine"},
                ),
                "resolution_x": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "resolution_y": ("INT", {"default": 512, "min": 256, "max": 4096, "step": 64}),
                "batch_size_override": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "gradient_accumulation_override": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "optimizer": (
                    ["AdamW", "AdamW8bit", "Lion", "PagedAdamW8bit"],
                    {"default": "AdamW8bit"},
                ),
                "loss_function": (
                    ["MSE", "Huber", "MAE"],
                    {"default": "MSE"},
                ),
                "seed": ("INT", {"default": 0, "min": 0, "max": 2147483647, "step": 1}),
                "control_after_generate": (
                    ["randomize", "fixed", "increment"],
                    {"default": "randomize"},
                ),
                "training_dtype": (DTYPE_CHOICES, {"default": "bf16"}),
                "lora_dtype": (DTYPE_CHOICES, {"default": "bf16"}),
                "quantized_backward": ("BOOLEAN", {"default": False}),
                "algorithm": (
                    ["LoRA", "LoCon", "LoHa"],
                    {"default": "LoRA"},
                ),
                "gradient_checkpointing": ("BOOLEAN", {"default": True}),
                "checkpoint_depth": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "offloading": ("BOOLEAN", {"default": False}),
                "existing_lora": (list_lora_files(), {"default": "[None]"}),
                "bucket_mode": ("BOOLEAN", {"default": False}),
                "bypass_mode": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "latents": ("LATENT",),
            },
        }

    def run(
        self,
        validated_dataset: Dict[str, Any],
        precision_plan: Dict[str, Any],
        preview_config: Dict[str, Any],
        rank: int,
        learning_rate: float,
        max_train_steps: int,
        lr_scheduler: str,
        resolution_x: int,
        resolution_y: int,
        batch_size_override: int,
        gradient_accumulation_override: int,
        optimizer: str,
        loss_function: str,
        seed: int,
        control_after_generate: str,
        training_dtype: str,
        lora_dtype: str,
        quantized_backward: bool,
        algorithm: str,
        gradient_checkpointing: bool,
        checkpoint_depth: int,
        offloading: bool,
        existing_lora: str,
        bucket_mode: bool,
        bypass_mode: bool,
        model: Any = None,
        positive: Any = None,
        latents: Any = None,
    ) -> Tuple[Dict[str, Any], Any, Dict[str, Any], int]:
        train_job = prepare_training_run(
            validated_dataset=validated_dataset,
            precision_plan=precision_plan,
            preview_config=preview_config,
            base_model_path=_default_base_model_path(model),
            output_dir=_default_output_dir(),
            output_name=_default_output_name(validated_dataset.get("dataset_dir", ""), "dynatrain_advanced_lora"),
            backend_workdir=_default_backend_workdir(),
            python_executable=sys.executable,
            entrypoint_mode="module",
            module_name="onetrainer",
            script_path="",
            network_dim=rank,
            network_alpha=rank,
            learning_rate=learning_rate,
            max_train_steps=max_train_steps,
            lr_scheduler=lr_scheduler,
            resolution_x=resolution_x,
            resolution_y=resolution_y,
            batch_size_override=batch_size_override,
            gradient_accumulation_override=gradient_accumulation_override,
            execute=False,
            extra_args="",
            runtime_root="",
            model=model,
            positive=positive,
            latents=latents,
            optimizer=optimizer,
            training_dtype=training_dtype,
            lora_dtype=lora_dtype,
            loss_function=loss_function,
            seed=seed,
            control_after_generate=control_after_generate,
            quantized_backward=quantized_backward,
            algorithm=algorithm,
            gradient_checkpointing=gradient_checkpointing,
            checkpoint_depth=checkpoint_depth,
            offloading=offloading,
            existing_lora=existing_lora,
            bucket_mode=bucket_mode,
            bypass_mode=bypass_mode,
        )
        lora_payload = build_lora_payload(train_job)
        loss_map = build_loss_map(train_job)
        preview_image = _resolve_preview_image(
            train_job.get("latest_preview", ""),
            "\n".join(preview_config.get("sample_prompts", [])),
        )
        return lora_payload, preview_image, loss_map, max_train_steps


class SaveLoRAWeightsNode:
    CATEGORY = "DynaTrain/Training"
    FUNCTION = "run"
    RETURN_TYPES = ("DYNA_LORA", "STRING", "STRING")
    RETURN_NAMES = ("lora", "saved_path", "summary")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora": ("DYNA_LORA",),
                "prefix": ("STRING", {"default": "loras/dynatrain_lora"}),
                "steps": ("INT", {"default": 0, "min": 0, "max": 1000000, "step": 1}),
            }
        }

    def run(self, lora: Dict[str, Any], prefix: str, steps: int) -> Tuple[Dict[str, Any], str, str]:
        saved_lora = save_lora_weights(lora, prefix=prefix, steps=steps)
        summary = _to_pretty_json(saved_lora)
        return saved_lora, saved_lora["saved_path"], summary


class PlotLossGraphNode:
    CATEGORY = "DynaTrain/Training"
    FUNCTION = "run"
    RETURN_TYPES = ("IMAGE", "STRING", "STRING")
    RETURN_NAMES = ("graph_image", "saved_path", "summary")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "loss": ("DYNA_LOSS_MAP",),
                "filename_prefix": ("STRING", {"default": "loss_graph"}),
            }
        }

    def run(self, loss: Dict[str, Any], filename_prefix: str) -> Tuple[Any, str, str]:
        points = loss_history(loss.get("log_path", ""))
        title = f"Loss Graph - {loss.get('job_id', 'unknown-job')}"
        image, comfy_image = _render_loss_graph(points, title)
        saved_path = _save_loss_graph_image(image, filename_prefix, loss)
        summary = _to_pretty_json(
            {
                "job_id": loss.get("job_id", ""),
                "points": len(points),
                "saved_path": saved_path,
                "latest_loss": loss.get("latest_loss"),
                "status": loss.get("status", ""),
            }
        )
        return comfy_image, saved_path, summary


class TrainingJobStatusNode:
    CATEGORY = "DynaTrain/Training"
    FUNCTION = "run"
    RETURN_TYPES = ("DYNA_TRAIN_JOB", "STRING", "IMAGE", "STRING", "STRING", "INT", "STRING", "FLOAT", "STRING", "STRING")
    RETURN_NAMES = (
        "train_job",
        "status",
        "preview_image",
        "log_tail",
        "preview_dir",
        "returncode",
        "latest_preview",
        "latest_loss",
        "latest_loss_line",
        "summary",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "train_job": ("DYNA_TRAIN_JOB",),
                "lora": ("DYNA_LORA",),
                "job_id": ("STRING", {"default": ""}),
                "runtime_root": ("STRING", {"default": ""}),
            },
        }

    def run(
        self,
        train_job: Dict[str, Any] | None = None,
        lora: Dict[str, Any] | None = None,
        job_id: str = "",
        runtime_root: str = "",
    ) -> Tuple[Dict[str, Any], str, Any, str, str, int, str, float, str, str]:
        resolved = get_job_status(train_job=train_job or lora, job_id=job_id, runtime_root=runtime_root)
        summary = _to_pretty_json(resolved)
        preview_image = _resolve_preview_image(
            resolved.get("latest_preview", ""),
            resolved.get("latest_loss_line", "") or "No generated preview yet.",
        )
        return (
            resolved,
            resolved["status"],
            preview_image,
            resolved.get("log_tail", ""),
            resolved.get("preview_dir", ""),
            resolved.get("returncode", -1) if resolved.get("returncode") is not None else -1,
            resolved.get("latest_preview", ""),
            float(resolved.get("latest_loss")) if resolved.get("latest_loss") is not None else -1.0,
            resolved.get("latest_loss_line", ""),
            summary,
        )


NODE_CLASS_MAPPINGS = {
    "DynaTrainDatasetSanityCheck": DatasetSanityCheckNode,
    "DynaTrainPrecisionPlanner": PrecisionPlannerNode,
    "DynaTrainCaptionPairValidator": CaptionPairValidatorNode,
    "DynaTrainSamplePreviewDuringTraining": SamplePreviewDuringTrainingNode,
    "DynaTrainTrainLoRA": TrainLoRANode,
    "DynaTrainTrainLoRAAdvanced": TrainLoRAAdvancedNode,
    "DynaTrainSaveLoRAWeights": SaveLoRAWeightsNode,
    "DynaTrainPlotLossGraph": PlotLossGraphNode,
    "DynaTrainTrainingJobStatus": TrainingJobStatusNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DynaTrainDatasetSanityCheck": "Dataset Sanity Check",
    "DynaTrainPrecisionPlanner": "Precision Planner",
    "DynaTrainCaptionPairValidator": "Caption Pair Validator",
    "DynaTrainSamplePreviewDuringTraining": "Sample Preview During Training",
    "DynaTrainTrainLoRA": "Train LoRA",
    "DynaTrainTrainLoRAAdvanced": "Train LoRA Advanced",
    "DynaTrainSaveLoRAWeights": "Save LoRA Weights",
    "DynaTrainPlotLossGraph": "Plot Loss Graph",
    "DynaTrainTrainingJobStatus": "Training Job Status",
}
