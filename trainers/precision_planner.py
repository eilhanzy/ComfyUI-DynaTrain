from __future__ import annotations

from typing import Any, Dict

from ..utils.config_presets import get_preset
from ..utils.vram_profiler import profile_vram_safety
from ..utils.warnings import merge_warnings


def plan_precision(
    *,
    model_family: str,
    vram_gb: float,
    requested_precision: str,
    save_precision: str,
    optimizer: str,
    gradient_accumulation_steps: int,
    gpu_supports_bf16: bool,
    allow_experimental_fp8: bool,
) -> Dict[str, Any]:
    preset = get_preset(model_family=model_family, vram_gb=vram_gb)
    warnings = []

    train_precision = requested_precision
    if requested_precision == "auto":
        train_precision = "bf16" if gpu_supports_bf16 and vram_gb >= 16 else "fp16"

    if train_precision == "bf16" and not gpu_supports_bf16:
        warnings.append("BF16 was requested but GPU support is disabled, so the plan falls back to FP16.")
        train_precision = "fp16"

    if train_precision == "fp8":
        if allow_experimental_fp8:
            warnings.append("FP8 is still experimental and should be treated as unstable for production training.")
        else:
            warnings.append("FP8 was requested without enabling experimental mode, so the plan falls back to FP16.")
            train_precision = "fp16"

    if train_precision == "nvfp4":
        warnings.append(
            "NVFP4 is hardware- and backend-dependent. Confirm support on your GPU and trainer before relying on it."
        )

    chosen_save_precision = save_precision
    if save_precision == "auto":
        chosen_save_precision = train_precision

    if chosen_save_precision == "fp8":
        warnings.append("Saving LoRA weights in FP8 depends on backend support and should be validated on a short run.")

    if chosen_save_precision == "nvfp4":
        warnings.append("Saving LoRA weights in NVFP4 is experimental and depends on matching backend support.")

    if optimizer == "lion":
        warnings.append("Lion support is enabled as an optional path; verify convergence on a short run before full training.")

    if optimizer == "paged_adamw8bit":
        warnings.append("Paged optimizers are treated as roadmap-grade support and may depend on your installed backend.")

    vram_profile = profile_vram_safety(
        model_family=model_family,
        vram_gb=vram_gb,
        resolution=preset["resolution"],
        batch_size=preset["batch_size"],
        gradient_accumulation_steps=gradient_accumulation_steps,
        train_precision=train_precision,
    )
    warnings = merge_warnings(warnings, vram_profile["warnings"])

    return {
        "model_family": model_family,
        "vram_gb": vram_gb,
        "requested_precision": requested_precision,
        "train_precision": train_precision,
        "save_precision": chosen_save_precision,
        "optimizer": optimizer,
        "optimizer_backend_name": _normalize_optimizer_name(optimizer),
        "preset": preset,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "vram_profile": vram_profile,
        "backend_config_defaults": {
            "train_precision": train_precision,
            "save_precision": chosen_save_precision,
            "optimizer": _normalize_optimizer_name(optimizer),
            "batch_size": preset["batch_size"],
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "resolution": preset["resolution"],
        },
        "warnings": warnings,
    }


def _normalize_optimizer_name(optimizer: str) -> str:
    mapping = {
        "adamw8bit": "AdamW8bit",
        "adamw": "AdamW",
        "lion": "Lion",
        "paged_adamw8bit": "PagedAdamW8bit",
    }
    return mapping[optimizer]
