from __future__ import annotations

import json
import os
import shlex
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List
from uuid import uuid4

from ..utils.comfy_paths import resolve_lora_path
from ..utils.config_presets import get_preset
from ..utils.runtime import (
    build_job_artifact_paths,
    default_runtime_root,
    latest_loss,
    latest_preview_path,
    log_tail,
    read_metadata,
    write_metadata,
)
from ..utils.warnings import merge_warnings
from .preview_runner import merge_preview_sources

OPTIMIZER_NAME_MAP = {
    "adamw8bit": "AdamW8bit",
    "adamw": "AdamW",
    "lion": "Lion",
    "paged_adamw8bit": "PagedAdamW8bit",
    "AdamW8bit": "AdamW8bit",
    "AdamW": "AdamW",
    "Lion": "Lion",
    "PagedAdamW8bit": "PagedAdamW8bit",
}


def _ensure_required_text(value: str, label: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{label} is required.")
    return normalized


def _normalize_optimizer_name(optimizer: str) -> str:
    normalized = optimizer.strip()
    if normalized not in OPTIMIZER_NAME_MAP:
        raise ValueError(f"Unsupported optimizer: {optimizer}")
    return OPTIMIZER_NAME_MAP[normalized]


def _resolve_dtype(value: str, fallback: str) -> str:
    normalized = value.strip().lower()
    return fallback if not normalized or normalized == "auto" else normalized


def _expected_weights_path(output_dir: str, output_name: str) -> str:
    return str(Path(output_dir) / f"{output_name}.safetensors")


def _resolve_runtime_settings(
    validated_dataset: Dict[str, Any],
    precision_plan: Dict[str, Any],
    resolution_x: int,
    resolution_y: int,
    batch_size_override: int,
    gradient_accumulation_override: int,
) -> Dict[str, Any]:
    preset = get_preset(
        model_family=precision_plan["model_family"],
        vram_gb=precision_plan["vram_gb"],
    )
    batch_size = batch_size_override or preset["batch_size"]
    accumulation_steps = gradient_accumulation_override or preset["gradient_accumulation_steps"]
    default_resolution = preset["resolution"]
    resolved_resolution_x = resolution_x or default_resolution
    resolved_resolution_y = resolution_y or resolved_resolution_x
    return {
        "resolution": {"x": resolved_resolution_x, "y": resolved_resolution_y},
        "resolution_x": resolved_resolution_x,
        "resolution_y": resolved_resolution_y,
        "batch_size": batch_size,
        "gradient_accumulation_steps": accumulation_steps,
        "caption_extension": validated_dataset["caption_extension"],
        "cache_latents": preset["cache_latents"],
        "cache_latents_to_disk": preset["cache_latents_to_disk"],
        "gradient_checkpointing": preset["gradient_checkpointing"],
        "use_xformers": preset["use_xformers"],
    }


def _build_entrypoint(
    *,
    python_executable: str,
    entrypoint_mode: str,
    module_name: str,
    script_path: str,
) -> List[str]:
    command = [_ensure_required_text(python_executable, "Python executable")]
    if entrypoint_mode == "module":
        command.extend(["-m", _ensure_required_text(module_name, "Module name")])
        return command
    if entrypoint_mode == "script":
        command.append(_ensure_required_text(script_path, "Script path"))
        return command
    raise ValueError(f"Unsupported entrypoint mode: {entrypoint_mode}")


def _build_backend_command(
    *,
    python_executable: str,
    entrypoint_mode: str,
    module_name: str,
    script_path: str,
    config_path: str,
    extra_args: str,
) -> List[str]:
    command = _build_entrypoint(
        python_executable=python_executable,
        entrypoint_mode=entrypoint_mode,
        module_name=module_name,
        script_path=script_path,
    )
    command.extend(["--config", config_path])
    if extra_args.strip():
        command.extend(shlex.split(extra_args.strip()))
    return command


def _render_command(command: List[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def _python_executable_exists(python_executable: str) -> bool:
    executable = python_executable.strip()
    if not executable:
        return False
    if Path(executable).is_absolute():
        return Path(executable).exists()
    return shutil.which(executable) is not None


def _build_backend_missing_message(
    *,
    python_executable: str,
    entrypoint_mode: str,
    module_name: str,
    script_path: str,
) -> str:
    if entrypoint_mode == "module":
        return (
            f"OneTrainer backend module '{module_name}' was not found for python executable "
            f"'{python_executable}'. Install OneTrainer in that environment or keep execute disabled "
            "and use dry-run mode."
        )
    return (
        f"OneTrainer backend script was not found at '{script_path}'. Point script_path to a valid "
        "backend entrypoint or keep execute disabled and use dry-run mode."
    )


def _inspect_backend_entrypoint(
    *,
    python_executable: str,
    entrypoint_mode: str,
    module_name: str,
    script_path: str,
) -> Dict[str, str | bool]:
    if not _python_executable_exists(python_executable):
        return {
            "backend_ready": False,
            "backend_check_message": (
                f"Python executable '{python_executable}' was not found. Point the node to a valid interpreter or use dry-run mode."
            ),
        }

    if entrypoint_mode == "script":
        normalized_script = script_path.strip()
        if Path(normalized_script).exists():
            return {
                "backend_ready": True,
                "backend_check_message": f"Backend script is available at '{normalized_script}'.",
            }
        return {
            "backend_ready": False,
            "backend_check_message": _build_backend_missing_message(
                python_executable=python_executable,
                entrypoint_mode=entrypoint_mode,
                module_name=module_name,
                script_path=script_path,
            ),
        }

    normalized_module_name = module_name.strip()
    probe = subprocess.run(
        [
            python_executable,
            "-c",
            (
                "import importlib.util, sys; "
                f"sys.exit(0 if importlib.util.find_spec({normalized_module_name!r}) else 1)"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode == 0:
        return {
            "backend_ready": True,
            "backend_check_message": (
                f"Backend module '{normalized_module_name}' is importable in python executable '{python_executable}'."
            ),
        }
    return {
        "backend_ready": False,
        "backend_check_message": _build_backend_missing_message(
            python_executable=python_executable,
            entrypoint_mode=entrypoint_mode,
            module_name=module_name,
            script_path=script_path,
        ),
    }


def _validate_backend_entrypoint(
    *,
    python_executable: str,
    entrypoint_mode: str,
    module_name: str,
    script_path: str,
) -> None:
    inspection = _inspect_backend_entrypoint(
        python_executable=python_executable,
        entrypoint_mode=entrypoint_mode,
        module_name=module_name,
        script_path=script_path,
    )
    if not inspection["backend_ready"]:
        raise FileNotFoundError(str(inspection["backend_check_message"]))


def _build_onetrainer_config(
    *,
    job_id: str,
    validated_dataset: Dict[str, Any],
    precision_plan: Dict[str, Any],
    preview_config: Dict[str, Any],
    base_model_path: str,
    output_dir: str,
    output_name: str,
    runtime_settings: Dict[str, Any],
    network_dim: int,
    network_alpha: int,
    learning_rate: float,
    max_train_steps: int,
    lr_scheduler: str,
    preview_dir: str,
    optimizer_backend_name: str,
    training_dtype: str,
    lora_dtype: str,
    loss_function: str,
    seed: int,
    control_after_generate: str,
    quantized_backward: bool,
    algorithm: str,
    checkpoint_depth: int,
    offloading: bool,
    existing_lora: str,
    existing_lora_path: str,
    bucket_mode: bool,
    bypass_mode: bool,
) -> Dict[str, Any]:
    return {
        "format": "dynatrain.onetrainer.v1",
        "job_id": job_id,
        "paths": {
            "base_model_path": base_model_path,
            "dataset_dir": validated_dataset["dataset_dir"],
            "output_dir": output_dir,
            "output_name": output_name,
            "preview_dir": preview_dir,
            "existing_lora_path": existing_lora_path,
        },
        "dataset": {
            "image_count": validated_dataset["image_count"],
            "caption_extension": validated_dataset["caption_extension"],
            "image_paths": validated_dataset["image_paths"],
            "duplicate_groups": validated_dataset["duplicate_groups"],
            "repeated_caption_groups": validated_dataset["repeated_caption_groups"],
        },
        "training": {
            "network_dim": network_dim,
            "network_alpha": network_alpha,
            "learning_rate": learning_rate,
            "max_train_steps": max_train_steps,
            "lr_scheduler": lr_scheduler,
            "resolution": runtime_settings["resolution"],
            "resolution_x": runtime_settings["resolution_x"],
            "resolution_y": runtime_settings["resolution_y"],
            "batch_size": runtime_settings["batch_size"],
            "gradient_accumulation_steps": runtime_settings["gradient_accumulation_steps"],
            "cache_latents": runtime_settings["cache_latents"],
            "cache_latents_to_disk": runtime_settings["cache_latents_to_disk"],
            "gradient_checkpointing": runtime_settings["gradient_checkpointing"],
            "use_xformers": runtime_settings["use_xformers"],
            "optimizer": optimizer_backend_name,
            "training_dtype": training_dtype,
            "lora_dtype": lora_dtype,
            "loss_function": loss_function,
            "seed": seed,
            "control_after_generate": control_after_generate,
            "quantized_backward": quantized_backward,
            "algorithm": algorithm,
            "checkpoint_depth": checkpoint_depth,
            "offloading": offloading,
            "bucket_mode": bucket_mode,
            "bypass_mode": bypass_mode,
        },
        "lora": {
            "rank": network_dim,
            "alpha": network_alpha,
            "algorithm": algorithm,
            "existing_lora": existing_lora,
            "existing_lora_path": existing_lora_path,
            "dtype": lora_dtype,
        },
        "precision": {
            "requested_precision": precision_plan["requested_precision"],
            "train_precision": training_dtype,
            "save_precision": lora_dtype,
            "optimizer": optimizer_backend_name,
            "optimizer_backend_name": optimizer_backend_name,
            "vram_gb": precision_plan["vram_gb"],
            "vram_profile": precision_plan["vram_profile"],
        },
        "preview": preview_config,
    }


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _prepare_job_payload(
    *,
    job_id: str,
    metadata_path: Path,
    config_path: Path,
    log_path: Path,
    preview_dir: Path,
    backend_workdir: str,
    backend_command: List[str],
    runtime_settings: Dict[str, Any],
    warnings: List[str],
    status: str,
    steps: int,
    expected_weights_path: str,
    existing_lora: str,
    existing_lora_path: str,
    optimizer_backend_name: str,
    training_dtype: str,
    lora_dtype: str,
    algorithm: str,
) -> Dict[str, Any]:
    return {
        "job_id": job_id,
        "status": status,
        "created_at": _now_utc(),
        "started_at": None,
        "finished_at": None,
        "pid": None,
        "worker_pid": None,
        "returncode": None,
        "backend_workdir": backend_workdir,
        "command_list": backend_command,
        "command": _render_command(backend_command),
        "metadata_path": str(metadata_path),
        "config_path": str(config_path),
        "log_path": str(log_path),
        "preview_dir": str(preview_dir),
        "latest_preview": "",
        "runtime_settings": runtime_settings,
        "warnings": warnings,
        "steps": steps,
        "expected_weights_path": expected_weights_path,
        "existing_lora": existing_lora,
        "existing_lora_path": existing_lora_path,
        "optimizer": optimizer_backend_name,
        "training_dtype": training_dtype,
        "lora_dtype": lora_dtype,
        "algorithm": algorithm,
    }


def _launch_background_worker(job: Dict[str, Any]) -> int:
    worker_script = Path(__file__).with_name("background_job_runner.py")
    worker_command = [
        sys.executable,
        str(worker_script),
        "--metadata-path",
        job["metadata_path"],
    ]
    return os.spawnv(os.P_NOWAIT, sys.executable, worker_command)


def _resolve_paths(
    *,
    output_dir: str,
    output_name: str,
    runtime_root: str,
    job_id: str,
) -> Dict[str, Path]:
    runtime_path = Path(runtime_root) if runtime_root.strip() else default_runtime_root()
    return build_job_artifact_paths(
        runtime_root=runtime_path,
        output_dir=Path(output_dir),
        output_name=output_name,
        job_id=job_id,
    )


def prepare_training_run(
    *,
    validated_dataset: Dict[str, Any],
    precision_plan: Dict[str, Any],
    preview_config: Dict[str, Any],
    base_model_path: str,
    output_dir: str,
    output_name: str,
    backend_workdir: str,
    python_executable: str,
    entrypoint_mode: str,
    module_name: str,
    script_path: str,
    network_dim: int,
    network_alpha: int,
    learning_rate: float,
    max_train_steps: int,
    lr_scheduler: str,
    resolution_x: int,
    resolution_y: int,
    batch_size_override: int,
    gradient_accumulation_override: int,
    execute: bool,
    extra_args: str,
    runtime_root: str = "",
    model: Any = None,
    positive: Any = None,
    latents: Any = None,
    optimizer: str = "AdamW8bit",
    training_dtype: str = "auto",
    lora_dtype: str = "auto",
    loss_function: str = "MSE",
    seed: int = 0,
    control_after_generate: str = "randomize",
    quantized_backward: bool = False,
    algorithm: str = "LoRA",
    gradient_checkpointing: bool | None = None,
    checkpoint_depth: int = 1,
    offloading: bool = False,
    existing_lora: str = "[None]",
    bucket_mode: bool = False,
    bypass_mode: bool = False,
) -> Dict[str, Any]:
    if validated_dataset.get("blocked"):
        raise ValueError(
            "Dataset validation blocked training. Resolve validator errors before running Train LoRA Advanced."
        )

    dataset_dir = _ensure_required_text(validated_dataset["dataset_dir"], "Validated dataset dir")
    base_model_path = _ensure_required_text(base_model_path, "Base model path")
    output_dir = _ensure_required_text(output_dir, "Output dir")
    backend_workdir = _ensure_required_text(backend_workdir, "Backend workdir")
    output_name = _ensure_required_text(output_name, "Output name")
    job_id = f"{output_name}-{uuid4().hex[:12]}"
    optimizer_backend_name = _normalize_optimizer_name(optimizer)
    resolved_training_dtype = _resolve_dtype(training_dtype, precision_plan["train_precision"])
    resolved_lora_dtype = _resolve_dtype(lora_dtype, precision_plan["save_precision"])
    existing_lora_path = resolve_lora_path(existing_lora)

    preview_config = merge_preview_sources(
        preview_config,
        model=model,
        positive=positive,
        latents=latents,
    )

    runtime_settings = _resolve_runtime_settings(
        validated_dataset=validated_dataset,
        precision_plan=precision_plan,
        resolution_x=resolution_x,
        resolution_y=resolution_y,
        batch_size_override=batch_size_override,
        gradient_accumulation_override=gradient_accumulation_override,
    )
    if gradient_checkpointing is not None:
        runtime_settings["gradient_checkpointing"] = gradient_checkpointing

    warnings = merge_warnings(validated_dataset.get("warnings"), precision_plan.get("warnings"))
    paths = _resolve_paths(
        output_dir=output_dir,
        output_name=output_name,
        runtime_root=runtime_root,
        job_id=job_id,
    )
    expected_weights_path = _expected_weights_path(output_dir, output_name)
    config_payload = _build_onetrainer_config(
        job_id=job_id,
        validated_dataset={**validated_dataset, "dataset_dir": dataset_dir},
        precision_plan=precision_plan,
        preview_config=preview_config,
        base_model_path=base_model_path,
        output_dir=output_dir,
        output_name=output_name,
        runtime_settings=runtime_settings,
        network_dim=network_dim,
        network_alpha=network_alpha,
        learning_rate=learning_rate,
        max_train_steps=max_train_steps,
        lr_scheduler=lr_scheduler,
        preview_dir=str(paths["preview_dir"]),
        optimizer_backend_name=optimizer_backend_name,
        training_dtype=resolved_training_dtype,
        lora_dtype=resolved_lora_dtype,
        loss_function=loss_function,
        seed=seed,
        control_after_generate=control_after_generate,
        quantized_backward=quantized_backward,
        algorithm=algorithm,
        checkpoint_depth=checkpoint_depth,
        offloading=offloading,
        existing_lora=existing_lora,
        existing_lora_path=existing_lora_path,
        bucket_mode=bucket_mode,
        bypass_mode=bypass_mode,
    )
    backend_command = _build_backend_command(
        python_executable=python_executable,
        entrypoint_mode=entrypoint_mode,
        module_name=module_name,
        script_path=script_path,
        config_path=str(paths["config_path"]),
        extra_args=extra_args,
    )

    job = _prepare_job_payload(
        job_id=job_id,
        metadata_path=paths["metadata_path"],
        config_path=paths["config_path"],
        log_path=paths["log_path"],
        preview_dir=paths["preview_dir"],
        backend_workdir=backend_workdir,
        backend_command=backend_command,
        runtime_settings=runtime_settings,
        warnings=warnings,
        status="dry_run",
        steps=max_train_steps,
        expected_weights_path=expected_weights_path,
        existing_lora=existing_lora,
        existing_lora_path=existing_lora_path,
        optimizer_backend_name=optimizer_backend_name,
        training_dtype=resolved_training_dtype,
        lora_dtype=resolved_lora_dtype,
        algorithm=algorithm,
    )
    job["dataset_dir"] = dataset_dir
    job["entrypoint_mode"] = entrypoint_mode
    job["module_name"] = module_name.strip()
    job["script_path"] = script_path.strip()
    job["python_executable"] = python_executable.strip()
    job["config_payload"] = config_payload
    job.update(
        _inspect_backend_entrypoint(
            python_executable=python_executable,
            entrypoint_mode=entrypoint_mode,
            module_name=module_name,
            script_path=script_path,
        )
    )

    if not execute:
        return enrich_job_status(job)

    _validate_backend_entrypoint(
        python_executable=python_executable,
        entrypoint_mode=entrypoint_mode,
        module_name=module_name,
        script_path=script_path,
    )
    paths["metadata_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["log_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["preview_dir"].mkdir(parents=True, exist_ok=True)
    paths["config_path"].parent.mkdir(parents=True, exist_ok=True)
    paths["config_path"].write_text(json.dumps(config_payload, indent=2, ensure_ascii=True), encoding="utf-8")

    job["status"] = "queued"
    write_metadata(paths["metadata_path"], job)
    worker_pid = _launch_background_worker(job)
    persisted = read_metadata(paths["metadata_path"])
    persisted["worker_pid"] = worker_pid
    write_metadata(paths["metadata_path"], persisted)
    return enrich_job_status(persisted)


def enrich_job_status(job: Dict[str, Any], max_log_chars: int = 4000) -> Dict[str, Any]:
    enriched = dict(job)
    enriched["latest_preview"] = latest_preview_path(enriched.get("preview_dir", ""))
    enriched["log_tail"] = log_tail(enriched.get("log_path", ""), max_chars=max_log_chars)
    enriched.update(latest_loss(enriched.get("log_path", "")))
    return enriched


def build_lora_payload(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "name": Path(job.get("expected_weights_path", "")).stem or job["job_id"],
        "weights_path": job.get("expected_weights_path", ""),
        "existing_lora": job.get("existing_lora", "[None]"),
        "existing_lora_path": job.get("existing_lora_path", ""),
        "metadata_path": job.get("metadata_path", ""),
        "config_path": job.get("config_path", ""),
        "log_path": job.get("log_path", ""),
        "preview_dir": job.get("preview_dir", ""),
        "status": job.get("status", "dry_run"),
        "steps": job.get("steps", 0),
        "optimizer": job.get("optimizer", ""),
        "training_dtype": job.get("training_dtype", ""),
        "lora_dtype": job.get("lora_dtype", ""),
        "algorithm": job.get("algorithm", "LoRA"),
        "backend_ready": job.get("backend_ready", False),
        "backend_check_message": job.get("backend_check_message", ""),
    }


def build_loss_map(job: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "job_id": job["job_id"],
        "metadata_path": job.get("metadata_path", ""),
        "log_path": job.get("log_path", ""),
        "status": job.get("status", "dry_run"),
        "latest_loss": job.get("latest_loss"),
        "latest_loss_line": job.get("latest_loss_line", ""),
    }


def get_job_status(*, train_job: Dict[str, Any] | None = None, job_id: str = "", runtime_root: str = "") -> Dict[str, Any]:
    if train_job:
        metadata_path = train_job.get("metadata_path", "")
        if metadata_path:
            metadata_file = Path(metadata_path)
            if metadata_file.exists():
                job = read_metadata(metadata_file)
                return enrich_job_status(job)
        return enrich_job_status(train_job)

    normalized_job_id = job_id.strip()
    if not normalized_job_id:
        raise ValueError("Training Job Status requires either a train job payload or a job_id.")

    runtime_path = Path(runtime_root) if runtime_root.strip() else default_runtime_root()
    metadata_path = runtime_path / "jobs" / f"{normalized_job_id}.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Job metadata was not found for job_id '{normalized_job_id}'.")

    job = read_metadata(metadata_path)
    return enrich_job_status(job)
