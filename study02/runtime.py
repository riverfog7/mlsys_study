"""Fresh-process experiment lifecycle. Only stdlib is imported before setup."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import traceback
import uuid

STUDY_DIR = Path(__file__).resolve().parent
REPO_DIR = STUDY_DIR.parent


def now():
    return datetime.now(timezone.utc).isoformat()


def new_run_dir(name):
    return STUDY_DIR / "artifacts" / f"{name}-{uuid.uuid4().hex[:12]}"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def configure_environment(out, name):
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    settings = {
        "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1",
        "TORCHINDUCTOR_COMPILE_THREADS": "1",
        "TORCH_COMPILE_DEBUG": "1" if name in {"inductor", "profiling"} else "0",
        "TORCH_COMPILE_DEBUG_DIR": str(raw / "debug"),
        "TORCHINDUCTOR_CACHE_DIR": str(raw / "inductor_cache"),
        "TRITON_CACHE_DIR": str(raw / "triton_cache"),
        "TRITON_KERNEL_DUMP": "1", "TRITON_DUMP_DIR": str(raw / "triton_dump"),
        "TORCH_LOGS": "graph_breaks,recompiles",
    }
    os.environ.update(settings)
    return settings


def cuda_environment():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError(
            "이 실습은 NVIDIA CUDA가 필요합니다. uv 환경의 PyTorch CUDA build와 "
            "GPU/driver를 확인하세요. CPU 결과로 대체하지 않습니다.")
    p = torch.cuda.get_device_properties(0)
    driver = None
    if shutil.which("nvidia-smi"):
        try:
            process = subprocess.run(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                                     capture_output=True, text=True, timeout=10, check=True)
            driver = process.stdout.strip().splitlines()[0]
        except (subprocess.SubprocessError, IndexError):
            pass
    return {
        "python": platform.python_version(), "executable": sys.executable,
        "platform": platform.platform(), "torch": torch.__version__,
        "torch_git": torch.version.git_version, "cuda": torch.version.cuda, "driver": driver,
        "triton": importlib.metadata.version("triton"),
        "gpu": p.name, "compute_capability": [p.major, p.minor],
        "gpu_memory_bytes": p.total_memory,
        "tools": {name: shutil.which(name) for name in ("nvidia-smi", "nvdisasm", "cuobjdump", "ncu")},
    }


def experiment_main(name, run, configure=None):
    parser = argparse.ArgumentParser(description=run.__doc__ or name)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="float32")
    if configure:
        configure(parser)
    args = parser.parse_args()
    out = (args.output or new_run_dir(name)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Reusing a previous run is an error, including when launched directly by CLI.
    with (out / ".started").open("x", encoding="utf-8") as stream:
        stream.write(now())
    settings = configure_environment(out, name)
    manifest = {
        "run_id": out.name, "experiment": name, "pid": os.getpid(), "started_at": now(),
        "command": [sys.executable, "-m", f"study02.experiments.{name}", *sys.argv[1:]],
        "settings": settings, "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "status": "running",
    }
    write_json(out / "manifest.json", manifest)
    error = None
    try:
        manifest["environment"] = cuda_environment()
        import torch
        torch.set_num_threads(1)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        manifest["environment"].update(tf32=False, cpu_threads=torch.get_num_threads())
        if args.dtype == "bfloat16" and not torch.cuda.is_bf16_supported(including_emulation=False):
            raise RuntimeError("이 GPU는 native BF16을 지원하지 않습니다. float32 또는 float16을 선택하세요.")
        result = run(args, out)
        result.update(run_id=out.name, device="cuda", dtype=args.dtype)
        write_json(out / "result.json", result)
        manifest["status"] = "completed"
    except Exception as exc:
        error = exc
        manifest.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        (out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    finally:
        manifest["finished_at"] = now()
        manifest["source_hashes"] = {
            str(p.relative_to(STUDY_DIR)): digest(p) for p in STUDY_DIR.rglob("*.py")
            if "artifacts" not in p.relative_to(STUDY_DIR).parts
        }
        manifest["artifacts"] = [
            {"path": str(p.relative_to(out)), "bytes": p.stat().st_size, "sha256": digest(p)}
            for p in sorted(out.rglob("*")) if p.is_file()
            and p.name not in {"manifest.json", "stdout.log", "stderr.log", ".started"}
        ]
        write_json(out / "manifest.json", manifest)
    if error:
        raise error
    print(f"{name}: completed → {out}")
