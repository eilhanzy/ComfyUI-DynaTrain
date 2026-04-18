from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import List


LORA_EXTENSIONS = {".safetensors", ".pt", ".ckpt", ".bin"}


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _folder_paths_module():
    try:
        import folder_paths  # type: ignore

        return folder_paths
    except ImportError:
        return None


def get_loras_dir() -> Path:
    env_dir = os.getenv("DYNATRAIN_LORAS_DIR", "").strip()
    if env_dir:
        path = Path(env_dir).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path

    folder_paths = _folder_paths_module()
    if folder_paths is not None:
        if hasattr(folder_paths, "get_folder_paths"):
            paths = folder_paths.get_folder_paths("loras")
            if paths:
                path = Path(paths[0])
                path.mkdir(parents=True, exist_ok=True)
                return path
        if hasattr(folder_paths, "folder_names_and_paths"):
            values = folder_paths.folder_names_and_paths.get("loras")
            if values and values[0]:
                path = Path(values[0][0])
                path.mkdir(parents=True, exist_ok=True)
                return path

    fallback = _repo_root() / "loras"
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def list_lora_files() -> List[str]:
    root = get_loras_dir()
    files = [
        str(path.relative_to(root))
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.suffix.lower() in LORA_EXTENSIONS
    ]
    return ["[None]"] + files


def resolve_lora_path(name: str) -> str:
    normalized = name.strip()
    if not normalized or normalized == "[None]":
        return ""

    folder_paths = _folder_paths_module()
    if folder_paths is not None and hasattr(folder_paths, "get_full_path"):
        full_path = folder_paths.get_full_path("loras", normalized)
        if full_path:
            return str(Path(full_path))

    candidate = get_loras_dir() / normalized
    if candidate.exists():
        return str(candidate)
    return ""


def build_lora_destination(prefix: str, steps: int) -> Path:
    root = get_loras_dir()
    normalized = prefix.strip() or "dynatrain_lora"
    if normalized.startswith("loras/"):
        normalized = normalized[len("loras/") :]
    if normalized.startswith("/"):
        normalized = normalized[1:]

    destination = Path(normalized)
    if destination.suffix.lower() not in LORA_EXTENSIONS:
        suffix = f"_{steps}" if steps > 0 else ""
        destination = Path(f"{destination}{suffix}.safetensors")

    full_path = root / destination
    full_path.parent.mkdir(parents=True, exist_ok=True)
    return full_path


def copy_lora_file(source_path: str, destination_path: Path) -> str:
    source = Path(source_path)
    if not source.exists():
        raise FileNotFoundError(f"LoRA weights file was not found at '{source_path}'.")
    shutil.copy2(source, destination_path)
    return str(destination_path)
