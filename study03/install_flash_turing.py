#!/usr/bin/env python3
"""Install the pinned Turing kernel for head-dimension-64 causal forward only.

This study build supports causal FP16 forward inference, not training.
Original CUDA kernels are unchanged; non-causal, other head sizes and backward calls raise.
Run with the notebook's Python after `uv sync`; CUDA and a compatible C++ compiler
must already be installed. Sources and compilation stay outside the repository.
"""

import argparse
import importlib.metadata
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

URL = "https://github.com/ssiu/flash-attention-turing"
REVISION = "9ef98fcb506bb1e2fe3cece50935e2935bf6b124"
CUTLASS_REVISION = "df18f5e4f5de76bed8be1de8e4c245f2f5ec3020"
VERSION = "0.0.0+g9ef98fcb.hdim64causalfwd"


def run(*args, **kwargs):
    return subprocess.run([str(arg) for arg in args], check=True, **kwargs)


def output(*args):
    return run(*args, text=True, capture_output=True).stdout.strip()


def preflight():
    import torch

    if not torch.cuda.is_available() or torch.version.cuda is None:
        raise RuntimeError("A CUDA-enabled PyTorch and a visible Turing GPU are required.")
    if torch.cuda.get_device_capability() != (7, 5):
        raise RuntimeError("This build requires a Turing GPU (SM 7.5), such as RTX 2080 or T4.")
    env = os.environ.copy()
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")
    nvcc = str(Path(env["CUDA_HOME"]) / "bin/nvcc") if env.get("CUDA_HOME") else shutil.which("nvcc")
    if not nvcc or not Path(nvcc).is_file():
        raise RuntimeError("CUDA development toolkit missing: set CUDA_HOME to the toolkit containing bin/nvcc.")
    release = re.search(r"release (\d+)\.(\d+)", output(nvcc, "--version"))
    if not release:
        raise RuntimeError("Could not determine the CUDA compiler version.")
    cuda_version = tuple(map(int, release.groups()))
    if cuda_version[0] != int(torch.version.cuda.split(".")[0]):
        raise RuntimeError("nvcc and PyTorch must use the same CUDA major version.")
    compiler = env.get("CXX") or shutil.which("g++")
    if not compiler:
        raise RuntimeError("A compatible C++ compiler is required (g++).")
    gcc_major = int(output(compiler, "-dumpfullversion", "-dumpversion").split(".")[0])
    if cuda_version == (12, 0) and gcc_major > 12:
        if not all(shutil.which(name) for name in ("gcc-12", "g++-12")):
            raise RuntimeError("CUDA 12.0 requires GCC 12 or older; gcc-12 and g++-12 were not found.")
        env.update(CC=shutil.which("gcc-12"), CXX=shutil.which("g++-12"))
    else:
        env["CXX"] = compiler
    env.update(CUDA_HOME=str(Path(nvcc).resolve().parent.parent), MAX_JOBS="1")
    return env


def checkout(source):
    if not source.exists():
        source.parent.mkdir(parents=True, exist_ok=True)
        run("git", "clone", "--filter=blob:none", "--no-checkout", URL, source)
        run("git", "-C", source, "checkout", "--detach", REVISION)
    if output("git", "-C", source, "rev-parse", "HEAD") != REVISION:
        raise RuntimeError("Source checkout has a different revision; use a fresh source directory.")
    if not (source / "csrc/cutlass/.git").exists():
        run("git", "-C", source, "submodule", "update", "--init", "csrc/cutlass")
    if output("git", "-C", source / "csrc/cutlass", "rev-parse", "HEAD") != CUTLASS_REVISION:
        raise RuntimeError("CUTLASS revision differs from the pinned upstream submodule.")


def verify_kernels(source):
    for directory, revision, paths in (
        (source, REVISION, ["csrc/flash_attn"]),
        (source / "csrc/cutlass", CUTLASS_REVISION, ["."]),
    ):
        changed = output("git", "-C", directory, "diff", "--name-only", revision, "--", *paths)
        if changed:
            raise RuntimeError("Original CUDA/kernel sources have local changes: " + changed)


def write_if_changed(path, text):
    if not path.exists() or path.read_text() != text:
        path.write_text(text)


def configure(source):
    setup = output("git", "-C", source, "show", REVISION + ":setup.py") + "\n"
    setup = setup.replace('name="flash_attn_turing",', f'name="flash_attn_turing",\n    version="{VERSION}",', 1)
    sources = ["csrc/flash_attn/flash_api.cpp",
               "csrc/flash_attn/src/flash_fwd_hdim64_fp16_causal_sm75.cu",
               "csrc/flash_attn/src/study03_unsupported.cpp"]
    replacement = "sources=[" + ",\n                     ".join('"' + name + '"' for name in sources) + "],"
    setup, count = re.subn(r"sources=\[.*?\],", replacement, setup, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError("Pinned setup.py does not have the expected source list.")
    stubs = '#include "flash.h"\n#include <stdexcept>\n'
    for direction in ("fwd", "bwd"):
        for dimension in (64, 96, 128):
            for causal in ("false", "true"):
                if direction == "fwd" and dimension == 64 and causal == "true":
                    continue
                stubs += (
                    f"template<> void run_mha_{direction}_<{dimension}, {causal}>(Flash_{direction}_params &) "
                    '{ throw std::runtime_error("This study build supports head-64 causal forward only"); }\n'
                )
    write_if_changed(source / "setup.py", setup)
    write_if_changed(source / "csrc/flash_attn/src/study03_unsupported.cpp", stubs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, help="Reuse a checkout at the pinned commit.")
    args = parser.parse_args()
    env = preflight()
    source = (args.source_dir or Path.home() / ".cache/mlsys-study/flash-attention-turing" / REVISION).resolve()
    checkout(source)
    verify_kernels(source)
    configure(source)
    verify_kernels(source)
    uv = shutil.which("uv")
    installer = [uv, "pip", "install", "--python", sys.executable] if uv else [sys.executable, "-m", "pip", "install"]
    missing = []
    for package in ("setuptools", "ninja", "wheel"):
        try:
            importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            missing.append(package)
    if missing:
        run(*installer, "--no-deps", *missing, env=env)
    print(f"Building {VERSION}: original head-64 causal forward kernel, no backward/training.", flush=True)
    run(*installer, "--no-deps", "--no-build-isolation", source, env=env)
    if importlib.metadata.version("flash-attn-turing") != VERSION:
        raise RuntimeError("Installed extension version does not match the scoped study build.")
    print("Installed. Restart the notebook kernel before importing flash_attn_turing.")


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ImportError, OSError, subprocess.CalledProcessError) as error:
        sys.exit(f"Installation stopped: {error}")
