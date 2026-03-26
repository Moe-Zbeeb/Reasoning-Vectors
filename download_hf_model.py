#!/usr/bin/env python3
"""Download a Hugging Face model repository by model ID."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

try:
    from huggingface_hub import snapshot_download
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing dependency: huggingface_hub. Install it with `pip install huggingface_hub`."
    ) from exc


def default_output_dir(model_id: str) -> Path:
    safe_name = model_id.replace("/", "--")
    return Path("downloaded_models") / safe_name


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model repository by model ID."
    )
    parser.add_argument(
        "model_id",
        help="Model ID on Hugging Face, e.g. meta-llama/Llama-3.1-8B",
    )
    parser.add_argument(
        "--output-dir",
        help="Directory to download into. Defaults to downloaded_models/<model-id>.",
    )
    parser.add_argument(
        "--revision",
        default=None,
        help="Optional branch, tag, or commit revision to download.",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="Hugging Face token. Defaults to the HF_TOKEN environment variable.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.model_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        local_path = snapshot_download(
            repo_id=args.model_id,
            repo_type="model",
            local_dir=str(output_dir),
            revision=args.revision,
            token=args.token,
        )
    except Exception as exc:
        print(f"Failed to download {args.model_id}: {exc}", file=sys.stderr)
        return 1

    print(f"Downloaded {args.model_id} to: {local_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
