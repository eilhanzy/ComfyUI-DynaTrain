from __future__ import annotations

from typing import Any, Dict


PRESETS = {
    "sd15": {
        "small": {
            "resolution": 512,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
        "medium": {
            "resolution": 512,
            "batch_size": 2,
            "gradient_accumulation_steps": 2,
            "cache_latents": True,
            "cache_latents_to_disk": False,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
        "large": {
            "resolution": 768,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "cache_latents": True,
            "cache_latents_to_disk": False,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
    },
    "sdxl": {
        "small": {
            "resolution": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
        "medium": {
            "resolution": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
        "large": {
            "resolution": 1024,
            "batch_size": 2,
            "gradient_accumulation_steps": 1,
            "cache_latents": True,
            "cache_latents_to_disk": False,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
    },
    "future_large": {
        "small": {
            "resolution": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 8,
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
        "medium": {
            "resolution": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 4,
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
        "large": {
            "resolution": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 2,
            "cache_latents": True,
            "cache_latents_to_disk": False,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
    },
    "flux": {
        "small": {
            "resolution": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 10,
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
        "medium": {
            "resolution": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 6,
            "cache_latents": True,
            "cache_latents_to_disk": True,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
        "large": {
            "resolution": 1024,
            "batch_size": 1,
            "gradient_accumulation_steps": 3,
            "cache_latents": True,
            "cache_latents_to_disk": False,
            "gradient_checkpointing": True,
            "use_xformers": True,
        },
    },
}


def get_preset(model_family: str, vram_gb: float) -> Dict[str, Any]:
    family_presets = PRESETS[model_family]
    return family_presets[_select_vram_tier(vram_gb)]


def _select_vram_tier(vram_gb: float) -> str:
    if vram_gb < 8:
        return "small"
    if vram_gb < 16:
        return "medium"
    return "large"
