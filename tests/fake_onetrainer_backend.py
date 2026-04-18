from __future__ import annotations

import argparse
import json
import time
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fail", action="store_true")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    preview_dir = Path(config["paths"]["preview_dir"])
    preview_dir.mkdir(parents=True, exist_ok=True)
    preview_path = preview_dir / f"{config['job_id']}_preview.png"
    preview_path.write_bytes(b"fake-preview")
    output_dir = Path(config["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    weights_path = output_dir / f"{config['paths']['output_name']}.safetensors"
    weights_path.write_bytes(b"fake-lora-weights")

    print(f"Loaded config for {config['job_id']}")
    print("step=1 loss=0.4321")
    print(f"Preview dir: {preview_dir}")
    time.sleep(0.2)
    print("step=2 loss=0.1234")

    if args.fail:
        print("step=3 loss=0.9876")
        print("Intentional backend failure")
        return 3

    print("step=3 loss=0.0456")
    print("Training completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
