from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from .comfy_paths import build_lora_destination, copy_lora_file


def resolve_lora_source(lora_payload: Dict[str, Any]) -> str:
    expected = lora_payload.get("weights_path", "")
    if expected and Path(expected).exists():
        return expected

    existing = lora_payload.get("existing_lora_path", "")
    if existing and Path(existing).exists():
        return existing

    raise FileNotFoundError(
        "LoRA source weights were not found. Wait for training to produce weights or select an existing_lora from the loras folder."
    )


def save_lora_weights(lora_payload: Dict[str, Any], prefix: str, steps: int) -> Dict[str, Any]:
    destination = build_lora_destination(prefix=prefix, steps=steps)
    saved_path = copy_lora_file(resolve_lora_source(lora_payload), destination)
    updated = dict(lora_payload)
    updated["saved_path"] = saved_path
    updated["weights_path"] = saved_path
    return updated
