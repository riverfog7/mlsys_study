"""Execute the entire notebook in a fresh uv kernel and export verification evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time
import uuid

import nbformat
from nbclient import NotebookClient
from nbconvert import HTMLExporter
from jupyter_client import KernelManager
from jupyter_client.kernelspec import KernelSpecManager

from study02.runtime import REPO_DIR, STUDY_DIR, cuda_environment, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", choices=("root", "study02"), default="root")
    args = parser.parse_args()
    environment = cuda_environment()
    output = STUDY_DIR / "artifacts" / f"validation-{uuid.uuid4().hex[:12]}"
    output.mkdir(parents=True)
    # An ephemeral kernelspec pins the exact uv executable without modifying the
    # user's global Jupyter kernel registrations.
    kernel_dir = output / "kernels" / "study02-verify"
    kernel_dir.mkdir(parents=True)
    write_json(kernel_dir / "kernel.json", {
        "argv": [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"],
        "display_name": "study02 verification", "language": "python",
    })
    manager = KernelManager(kernel_name="study02-verify",
                            kernel_spec_manager=KernelSpecManager(kernel_dirs=[str(kernel_dir.parent)]))
    notebook = nbformat.read(STUDY_DIR / "main.ipynb", as_version=4)
    nbformat.validate(notebook)
    def progress(cell, cell_index, **kwargs):
        if cell.cell_type == "code":
            print(f"cell {cell_index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:100]}", flush=True)
    client = NotebookClient(notebook, km=manager, timeout=600, on_cell_start=progress,
                            resources={"metadata": {"path": str(REPO_DIR if args.cwd == 'root' else STUDY_DIR)}})
    started = time.perf_counter()
    try:
        client.execute()
    finally:
        nbformat.write(notebook, output / "main.executed.ipynb")
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    streams = "\n".join(o.get("text", "") for c in notebook.cells for o in c.get("outputs", []) if o.output_type == "stream")
    paths = sorted(set(re.findall(r"원본: (study02/artifacts/[\w-]+)", streams)))
    runs = []
    for relative in paths:
        manifest = json.loads((REPO_DIR / relative / "manifest.json").read_text())
        if manifest["status"] != "completed":
            raise AssertionError(f"Run did not complete: {relative}")
        if Path(manifest["environment"]["executable"]).resolve() != Path(sys.executable).resolve():
            raise AssertionError("Notebook and child interpreter differ")
        runs.append({"run_id": manifest["run_id"], "experiment": manifest["experiment"],
                     "path": relative, "arguments": manifest["arguments"]})
    errors = [o for c in notebook.cells for o in c.get("outputs", []) if o.output_type == "error"]
    plots = sum("application/vnd.plotly.v1+json" in o.get("data", {}) for c in notebook.cells for o in c.get("outputs", []))
    code_cells = [c for c in notebook.cells if c.cell_type == "code"]
    if errors or not all(c.execution_count is not None for c in code_cells):
        raise AssertionError("Notebook has errors or unexecuted cells")
    if len(runs) < 15 or plots < 30:
        raise AssertionError(f"Missing expected experiments/figures: {len(runs)} runs, {plots} figures")
    html, _ = HTMLExporter(template_name="lab").from_notebook_node(notebook)
    (output / "main.html").write_text(html, encoding="utf-8")
    report = {"status": "passed", "seconds": time.perf_counter() - started, "cwd": args.cwd,
              "code_cells": len(code_cells), "plotly_figures": plots, "environment": environment,
              "runs": runs, "notebook": str(output / "main.executed.ipynb"), "html": str(output / "main.html")}
    write_json(output / "verification.json", report)
    print(json.dumps({k: report[k] for k in ["status", "seconds", "code_cells", "plotly_figures", "html"]}, indent=2), flush=True)
    print(f"Evidence: {output}", flush=True)


if __name__ == "__main__":
    main()
