from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict


PREVIEW_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
LOSS_PATTERNS = [
    re.compile(r"\bloss\s*[:=]\s*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", re.IGNORECASE),
    re.compile(r"\bloss\b[^0-9-+]*([0-9]*\.?[0-9]+(?:[eE][+-]?[0-9]+)?)", re.IGNORECASE),
]


def default_runtime_root() -> Path:
    return Path(__file__).resolve().parent.parent / "runtime"


def build_job_artifact_paths(
    *,
    runtime_root: Path,
    output_dir: Path,
    output_name: str,
    job_id: str,
) -> Dict[str, Path]:
    return {
        "runtime_root": runtime_root,
        "metadata_path": runtime_root / "jobs" / f"{job_id}.json",
        "log_path": runtime_root / "logs" / f"{job_id}.log",
        "preview_dir": runtime_root / "previews" / job_id,
        "config_path": output_dir / f"{output_name}_{job_id}_onetrainer_config.json",
    }


def write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def read_metadata(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def latest_preview_path(preview_dir: str) -> str:
    if not preview_dir:
        return ""
    root = Path(preview_dir)
    if not root.exists():
        return ""
    previews = [
        path for path in root.iterdir() if path.is_file() and path.suffix.lower() in PREVIEW_EXTENSIONS
    ]
    if not previews:
        return ""
    latest = max(previews, key=lambda item: item.stat().st_mtime)
    return str(latest)


def log_tail(log_path: str, max_chars: int = 4000) -> str:
    if not log_path:
        return ""
    path = Path(log_path)
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="ignore")[-max_chars:]


def latest_loss(log_path: str) -> Dict[str, Any]:
    if not log_path:
        return {"latest_loss": None, "latest_loss_line": ""}

    path = Path(log_path)
    if not path.exists():
        return {"latest_loss": None, "latest_loss_line": ""}

    last_value = None
    last_line = ""
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        for pattern in LOSS_PATTERNS:
            match = pattern.search(line)
            if match:
                try:
                    last_value = float(match.group(1))
                    last_line = line.strip()
                except ValueError:
                    continue
    return {"latest_loss": last_value, "latest_loss_line": last_line}
