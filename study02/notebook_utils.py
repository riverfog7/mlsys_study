"""Notebook UI for isolated experiments. Never loads a previous run implicitly."""
from __future__ import annotations

import html
import inspect
import json
import os
from pathlib import Path
import signal
import subprocess
import sys

from study02.runtime import REPO_DIR, new_run_dir

EXPERIMENTS = {"python_runtime", "dynamo", "aot_autograd", "inductor", "profiling", "triton_codegen"}


def run_experiment(name, *, timeout=600, **options):
    if name not in EXPERIMENTS:
        raise ValueError(f"Unknown experiment: {name}")
    out = new_run_dir(name)
    out.mkdir(parents=True)
    command = [sys.executable, "-m", f"study02.experiments.{name}", "--output", str(out)]
    for key, value in options.items():
        if value is None or value is False:
            continue
        command.append("--" + key.replace("_", "-"))
        if value is not True:
            command.append(str(value))
    print(f"CUDA 실험 실행: {name} · {out.name}")
    with (out / "stdout.log").open("w") as stdout, (out / "stderr.log").open("w") as stderr:
        process = subprocess.Popen(command, cwd=REPO_DIR, stdout=stdout, stderr=stderr,
                                   start_new_session=True)
        try:
            code = process.wait(timeout=timeout)
        except (subprocess.TimeoutExpired, KeyboardInterrupt) as exc:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise TimeoutError(f"{name}: {timeout}초 제한. 중단된 run: {out}") from exc
    if code:
        error = out / "error.txt"
        detail = (error if error.exists() else out / "stderr.log").read_text(errors="replace")[-7000:]
        raise RuntimeError(f"{name} 실패 (exit={code}). 이전 결과를 사용하지 않습니다.\n{out}\n{detail}")
    manifest = json.loads((out / "manifest.json").read_text())
    result = json.loads((out / "result.json").read_text())
    if manifest["status"] != "completed" or result["run_id"] != out.name:
        raise RuntimeError("실행 결과와 manifest가 일치하지 않습니다.")
    result["_run_dir"] = str(out)
    result["_environment"] = manifest["environment"]
    print(f"완료 · {manifest['environment']['gpu']} · 원본: {out.relative_to(REPO_DIR)}")
    return result


def run_file(result, relative):
    root = Path(result["_run_dir"]).resolve()
    path = (root / relative).resolve()
    if root not in path.parents:
        raise ValueError("Only this run's artifacts may be read")
    return path


def details(title, text):
    from IPython.display import HTML, display
    display(HTML(f"<details><summary>{html.escape(title)}</summary>"
                 f"<pre style='max-height:420px;overflow:auto'>{html.escape(str(text))}</pre></details>"))


def show_source(function):
    source = inspect.getsource(function)
    details(f"실제 소스: {inspect.getsourcefile(function)} · {function.__name__}", source)


def show_artifact(result, relative, max_lines=180):
    path = run_file(result, relative)
    lines = path.read_text(errors="replace").splitlines()
    excerpt = "\n".join(f"{i + 1:4d}  {line}" for i, line in enumerate(lines[:max_lines]))
    if len(lines) > max_lines:
        excerpt += f"\n… 전체 {len(lines)}줄: {path}"
    details(f"이번 실행의 원본: {relative}", excerpt)


def show_summary(rows, title="확인한 결과"):
    from IPython.display import HTML, display
    body = "".join(f"<tr><th style='text-align:left;padding:8px'>{html.escape(str(k))}</th>"
                   f"<td style='padding:8px'>{html.escape(str(v))}</td></tr>" for k, v in rows.items())
    display(HTML(f"<b>{html.escape(title)}</b><table>{body}</table>"))
