from __future__ import annotations

from typing import Any, Dict, List


def _normalize_prompts(sample_prompts: str) -> List[str]:
    prompts = [line.strip() for line in sample_prompts.splitlines() if line.strip()]
    return prompts or ["portrait photo, soft cinematic light"]


def _conditioning_summary(conditioning: Any) -> Dict[str, Any]:
    if conditioning is None:
        return {"connected": False}
    summary: Dict[str, Any] = {
        "connected": True,
        "type": type(conditioning).__name__,
    }
    if hasattr(conditioning, "__len__"):
        try:
            summary["items"] = len(conditioning)
        except TypeError:
            pass
    return summary


def _latent_summary(latents: Any) -> Dict[str, Any]:
    if latents is None:
        return {"connected": False}

    summary: Dict[str, Any] = {
        "connected": True,
        "type": type(latents).__name__,
    }
    if isinstance(latents, dict):
        samples = latents.get("samples")
        if samples is not None:
            shape = getattr(samples, "shape", None)
            if shape is not None:
                summary["shape"] = tuple(shape)
    return summary


def _model_summary(model: Any) -> Dict[str, Any]:
    if model is None:
        return {"connected": False}
    return {
        "connected": True,
        "type": type(model).__name__,
    }


def summarize_preview_sources(
    *,
    model: Any = None,
    positive: Any = None,
    latents: Any = None,
) -> Dict[str, Any]:
    return {
        "model": _model_summary(model),
        "positive_conditioning": _conditioning_summary(positive),
        "latents": _latent_summary(latents),
    }


def merge_preview_sources(
    preview_config: Dict[str, Any],
    *,
    model: Any = None,
    positive: Any = None,
    latents: Any = None,
) -> Dict[str, Any]:
    merged = dict(preview_config)
    connected_inputs = dict(merged.get("connected_inputs", {}))
    if model is not None or "model" not in connected_inputs:
        connected_inputs["model"] = _model_summary(model)
    if positive is not None or "positive_conditioning" not in connected_inputs:
        connected_inputs["positive_conditioning"] = _conditioning_summary(positive)
    if latents is not None or "latents" not in connected_inputs:
        connected_inputs["latents"] = _latent_summary(latents)
    merged["connected_inputs"] = connected_inputs
    merged["uses_comfy_preview_inputs"] = any(
        entry.get("connected")
        for entry in merged["connected_inputs"].values()
    )
    return merged


def build_preview_config(
    *,
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
) -> Dict[str, Any]:
    prompts = _normalize_prompts(sample_prompts)
    preview_config = {
        "enabled": enabled and sample_every_n_steps > 0,
        "sample_every_n_steps": sample_every_n_steps,
        "sample_prompts": prompts,
        "sampler": sampler,
        "sample_steps": sample_steps,
        "cfg_scale": cfg_scale,
        "seed": seed,
    }
    return merge_preview_sources(
        preview_config,
        model=model,
        positive=positive,
        latents=latents,
    )
