from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from ..utils.warnings import merge_warnings


def validate_caption_pairs(
    *,
    dataset_report: Dict[str, Any],
    warn_repeated_caption_ratio: float = 0.3,
) -> Dict[str, Any]:
    dataset_dir = Path(dataset_report["dataset_dir"])
    caption_extension = dataset_report["caption_extension"]
    image_rel_paths = dataset_report["image_paths"]

    errors: List[str] = list(dataset_report.get("errors", []))
    warnings: List[str] = list(dataset_report.get("warnings", []))
    missing_captions: List[str] = []
    empty_captions: List[str] = []
    captions: List[str] = []

    for rel_path in image_rel_paths:
        image_path = dataset_dir / rel_path
        caption_path = image_path.with_suffix(caption_extension)
        if not caption_path.exists():
            missing_captions.append(rel_path)
            continue

        caption_text = caption_path.read_text(encoding="utf-8", errors="ignore").strip()
        if not caption_text:
            empty_captions.append(rel_path)
            continue
        captions.append(caption_text)

    if missing_captions or empty_captions:
        errors.append("Training is blocked because caption mapping is broken.")

    repeated_caption_groups = _find_repeated_captions(captions)
    repeated_caption_items = sum(group["count"] for group in repeated_caption_groups)
    repeated_ratio = (repeated_caption_items / len(image_rel_paths)) if image_rel_paths else 0.0
    if repeated_caption_groups and repeated_ratio >= warn_repeated_caption_ratio:
        warnings.append(
            f"Repeated captions exceed the warning threshold: {repeated_caption_items} repeated samples across {len(repeated_caption_groups)} caption groups."
        )

    warnings = merge_warnings(warnings)
    return {
        "dataset_dir": str(dataset_dir),
        "caption_extension": caption_extension,
        "image_count": dataset_report["image_count"],
        "image_paths": image_rel_paths,
        "missing_captions": missing_captions,
        "empty_captions": empty_captions,
        "repeated_caption_groups": repeated_caption_groups,
        "duplicate_groups": dataset_report["duplicate_groups"],
        "errors": errors,
        "warnings": warnings,
        "blocked": bool(errors),
    }


def _find_repeated_captions(captions: List[str]) -> List[Dict[str, Any]]:
    repeated = []
    for caption, count in Counter(captions).most_common():
        if count < 2:
            continue
        repeated.append({"caption": caption, "count": count})
    return repeated
