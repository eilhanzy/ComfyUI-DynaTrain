from __future__ import annotations

from typing import Any, Dict, List


SAFE_ACCUMULATION_LIMITS = {
    "sd15": 16,
    "sdxl": 12,
    "future_large": 8,
    "flux": 6,
}


def profile_vram_safety(
    *,
    model_family: str,
    vram_gb: float,
    resolution: int,
    batch_size: int,
    gradient_accumulation_steps: int,
    train_precision: str,
) -> Dict[str, Any]:
    warnings: List[str] = []
    safe_limit = SAFE_ACCUMULATION_LIMITS[model_family]

    if gradient_accumulation_steps > safe_limit:
        warnings.append(
            f"Gradient accumulation ({gradient_accumulation_steps}) exceeds the recommended safety limit ({safe_limit}) for {model_family}."
        )

    pressure_score = round(
        (resolution / 512.0) * max(batch_size, 1) * max(gradient_accumulation_steps, 1),
        2,
    )
    if model_family == "sdxl":
        pressure_score = round(pressure_score * 1.8, 2)
    if model_family == "future_large":
        pressure_score = round(pressure_score * 2.4, 2)
    if model_family == "flux":
        pressure_score = round(pressure_score * 2.8, 2)

    if vram_gb < 8 and train_precision == "bf16":
        warnings.append("BF16 on sub-8 GB VRAM is likely unstable; FP16 is safer for this budget.")

    return {
        "pressure_score": pressure_score,
        "safe_accumulation_limit": safe_limit,
        "warnings": warnings,
    }
