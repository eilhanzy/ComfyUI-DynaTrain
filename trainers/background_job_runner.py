from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_metadata(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_metadata(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")


def run_job(metadata_path: Path) -> int:
    metadata = _read_metadata(metadata_path)
    log_path = Path(metadata["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    Path(metadata["preview_dir"]).mkdir(parents=True, exist_ok=True)

    metadata["status"] = "running"
    metadata["started_at"] = _now_utc()
    _write_metadata(metadata_path, metadata)

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"[DynaTrain] Launching background job {metadata['job_id']}\n")
        handle.flush()
        try:
            process = subprocess.Popen(
                metadata["command_list"],
                cwd=metadata["backend_workdir"],
                stdout=handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                text=True,
            )
            metadata["pid"] = process.pid
            _write_metadata(metadata_path, metadata)
            returncode = process.wait()
        except Exception as exc:
            handle.write(f"[DynaTrain] Background runner failed: {exc}\n")
            handle.flush()
            metadata["status"] = "failed"
            metadata["returncode"] = -1
            metadata["finished_at"] = _now_utc()
            _write_metadata(metadata_path, metadata)
            return -1

    metadata["status"] = "completed" if returncode == 0 else "failed"
    metadata["returncode"] = returncode
    metadata["finished_at"] = _now_utc()
    _write_metadata(metadata_path, metadata)
    return returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-path", required=True)
    args = parser.parse_args()
    return run_job(Path(args.metadata_path))


if __name__ == "__main__":
    raise SystemExit(main())
