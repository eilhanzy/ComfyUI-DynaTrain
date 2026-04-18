from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .duplicate_checker import find_duplicate_groups

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
}


def _discover_images(dataset_dir: Path, recursive: bool) -> List[Path]:
    pattern = "**/*" if recursive else "*"
    images = [
        path
        for path in dataset_dir.glob(pattern)
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    return sorted(images)


def _relative_paths(base_dir: Path, paths: Iterable[Path]) -> List[str]:
    return [str(path.relative_to(base_dir)) for path in paths]


def summarize_dataset(dataset_dir: str, caption_extension: str = ".txt", recursive: bool = True) -> Dict[str, Any]:
    root = Path(dataset_dir).expanduser()
    errors: List[str] = []
    warnings: List[str] = []

    if not root.exists():
        raise ValueError(f"Dataset dir does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Dataset path is not a directory: {root}")

    normalized_caption_extension = caption_extension if caption_extension.startswith(".") else f".{caption_extension}"
    image_paths = _discover_images(root, recursive=recursive)
    duplicate_groups = find_duplicate_groups(image_paths)

    if not image_paths:
        errors.append("Dataset contains no supported image files.")
    if len(image_paths) == 1:
        errors.append("Training is blocked because the dataset contains only one image.")

    duplicate_count = sum(len(group) for group in duplicate_groups)
    duplicate_ratio = (duplicate_count / len(image_paths)) if image_paths else 0.0
    if duplicate_ratio >= 0.2:
        warnings.append(
            f"Duplicate-heavy dataset detected: {duplicate_count} of {len(image_paths)} images share identical file hashes."
        )

    summary = {
        "dataset_dir": str(root.resolve()),
        "caption_extension": normalized_caption_extension,
        "recursive": recursive,
        "image_count": len(image_paths),
        "image_paths": _relative_paths(root, image_paths),
        "duplicate_groups": [
            _relative_paths(root, group) for group in duplicate_groups if len(group) > 1
        ],
        "errors": errors,
        "warnings": warnings,
        "blocked": bool(errors),
    }
    return summary
