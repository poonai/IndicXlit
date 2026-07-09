#!/usr/bin/env python3
"""Probe TensorRT-LLM environment readiness for the IndicXlit port."""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def run_command(argv: list[str], timeout: int = 20) -> dict[str, object]:
    try:
        completed = subprocess.run(
            argv,
            check=False,
            text=True,
            capture_output=True,
            timeout=timeout,
        )
        return {
            "argv": argv,
            "returncode": completed.returncode,
            "stdout": completed.stdout[-4000:],
            "stderr": completed.stderr[-4000:],
        }
    except Exception as exc:
        return {
            "argv": argv,
            "returncode": None,
            "error": repr(exc),
        }


def module_info(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"available": False, "error": repr(exc)}

    return {
        "available": True,
        "version": getattr(module, "__version__", None),
        "file": getattr(module, "__file__", None),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe TensorRT-LLM runtime/tool availability.")
    parser.add_argument(
        "--output",
        default="inference experiment/trtllm_port/artifacts/trtllm_env_probe.json",
        help="JSON output path.",
    )
    args = parser.parse_args()

    import torch

    executable_dir = Path(sys.executable).resolve().parent
    trtllm_build = shutil.which("trtllm-build")
    if trtllm_build is None:
        adjacent_trtllm_build = executable_dir / "trtllm-build"
        if adjacent_trtllm_build.is_file():
            trtllm_build = str(adjacent_trtllm_build)
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "executable": sys.executable,
        "path": os.environ.get("PATH"),
        "ld_library_path": os.environ.get("LD_LIBRARY_PATH"),
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        },
        "modules": {
            "tensorrt": module_info("tensorrt"),
            "tensorrt_llm": module_info("tensorrt_llm"),
        },
        "commands": {
            "trtllm-build": trtllm_build,
            "trtllm-build_help": run_command([trtllm_build, "--help"]) if trtllm_build else None,
            "nvidia-smi": run_command(["nvidia-smi"], timeout=5) if shutil.which("nvidia-smi") else None,
            "nvcc": run_command(["nvcc", "--version"], timeout=5) if shutil.which("nvcc") else None,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    print(f"Wrote {output}")
    print(
        "tensorrt_llm="
        f"{report['modules']['tensorrt_llm']['available']} "
        f"trtllm-build={bool(trtllm_build)} "
        f"cuda={report['torch']['cuda_available']}"
    )
    return 0 if report["modules"]["tensorrt_llm"]["available"] and trtllm_build else 2


if __name__ == "__main__":
    raise SystemExit(main())
