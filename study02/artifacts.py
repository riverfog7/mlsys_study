"""Read only this execution's compiler artifacts. Unknown structures stay unknown."""
from __future__ import annotations

import ast
from pathlib import Path
import re


def scheduler_graph(path, stage):
    text = Path(path).read_text()
    nested = set(re.findall(r"^\w+\.snodes\[\d+\] =\s*\n(\w+):", text, re.M))
    headers = [m for m in re.finditer(r"^(\w+): (\w+)\(([^\n]*)\)", text, re.M)
               if m.group(1) not in nested]
    nodes = []
    for i, match in enumerate(headers):
        name, kind = match.group(1), match.group(2)
        body = text[match.start():headers[i + 1].start() if i + 1 < len(headers) else len(text)]
        def dependencies(field):
            line = re.search(rf"^{re.escape(name)}\.{field}\s*=\s*(.*?)(?=^\w+\.\w+\s*=|^\w+:|\Z)",
                             body, re.M | re.S)
            return re.findall(r"(?:MemoryDep|StarDep|WeakDep)\(['\"]([^'\"]+)", line.group(1)) if line else []
        domain = re.search(r"\.group\.iteration = (.*)", body)
        nodes.append({"id": name, "op": kind, "target": match.group(3),
                      "reads": sorted(set(dependencies("met_dependencies") + dependencies("unmet_dependencies"))),
                      "writes": dependencies("writes"),
                      "members": re.findall(r"op\d+", name),
                      "metadata": {"domain": "fused: see member domains" if "Fused" in kind else
                                   domain.group(1) if domain else "external/unknown"},
                      "source": body[:4000]})
    writers = {buffer: node["id"] for node in nodes for buffer in node["writes"]}
    edges = [{"source": writers[buffer], "target": node["id"], "buffer": buffer}
             for node in nodes for buffer in node["reads"] if buffer in writers and writers[buffer] != node["id"]]
    return {"stage": stage, "nodes": nodes, "edges": edges, "provenance": str(path)}


def wrapper_data(path):
    """Logical wrapper source-line lifetimes, not physical GPU execution times."""
    text = Path(path).read_text()
    tree = ast.parse(text)
    function = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "call"), None)
    if function is None:
        return {"status": "unrecognized", "calls": [], "buffers": []}
    calls, buffers = [], {}
    for node in ast.walk(function):
        if isinstance(node, ast.Call):
            callee = ast.unparse(node.func)
            if (callee.startswith(("extern_kernels.", "torch.ops.aten.")) or
                    (callee.endswith(".run") and any(s in callee for s in ("triton", "multi_kernel", "cuda")))):
                calls.append({"line": node.lineno, "name": callee,
                              "kind": "external" if callee.startswith(("extern_kernels.", "torch.ops.aten.")) else "generated",
                              "source": ast.get_source_segment(text, node)})
    # CUDA wrappers put allocations inside `with torch.cuda._DeviceGuard(...)`.
    assignments = sorted((n for n in ast.walk(function) if isinstance(n, ast.Assign)), key=lambda n: n.lineno)
    for statement in assignments:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if isinstance(target, ast.Name) and re.fullmatch(r"buf\d+", target.id):
                value = statement.value
                storage, relation, shape, nbytes = target.id, "unknown", None, None
                if isinstance(value, ast.Name) and value.id in buffers:
                    storage = buffers[value.id]["storage"]
                    relation = "alias/reuse"
                elif isinstance(value, ast.Call):
                    callee = ast.unparse(value.func)
                    if callee == "reinterpret_tensor" and isinstance(value.args[0], ast.Name):
                        storage = buffers.get(value.args[0].id, {}).get("storage", value.args[0].id)
                        relation = "reinterpret/reuse"
                    elif "empty_strided" in callee:
                        relation = "allocation"
                        try:
                            shape = list(ast.literal_eval(value.args[0]))
                            dtype = ast.unparse(value.args[2])
                            itemsize = {"torch.float32": 4, "torch.float16": 2, "torch.bfloat16": 2,
                                        "torch.int64": 8, "torch.bool": 1}[dtype]
                            import math
                            nbytes = math.prod(shape) * itemsize
                        except (ValueError, TypeError, KeyError, IndexError):
                            pass  # A symbolic layout has no inferred byte size.
                buffers[target.id] = {"name": target.id, "storage": storage, "relation": relation,
                                      "first_line": statement.lineno, "last_line": statement.lineno,
                                      "shape": shape, "logical_bytes": nbytes}
    for node in ast.walk(function):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Load, ast.Del)) and node.id in buffers:
            buffers[node.id]["last_line"] = max(buffers[node.id]["last_line"], node.lineno)
    return {"status": "parsed", "calls": sorted(calls, key=lambda c: c["line"]),
            "buffers": list(buffers.values()), "lifetime_unit": "wrapper source line, not GPU time"}


def collect_inductor(out):
    # Debug runs are unique per experiment; never search a sibling repo or old run.
    debug = out / "raw" / "debug"
    captures = []
    for pre in sorted(debug.rglob("ir_pre_fusion.txt")):
        directory = pre.parent
        post, code = directory / "ir_post_fusion.txt", directory / "output_code.py"
        if not post.exists() or not code.exists():
            raise RuntimeError(f"Incomplete current compiler capture: {directory}")
        captures.append({"directory": str(directory.relative_to(out)),
                         "pre": scheduler_graph(pre, "pre-fusion"),
                         "post": scheduler_graph(post, "post-fusion"),
                         "wrapper": wrapper_data(code),
                         "files": {name: str((directory / name).relative_to(out)) for name in
                                   ("ir_pre_fusion.txt", "ir_post_fusion.txt", "output_code.py", "fx_graph_readable.py", "fx_graph_transformed.py")
                                   if (directory / name).exists()}})
    if not captures:
        raise RuntimeError("This run produced no Inductor debug IR. See stderr.log; no stored result is substituted.")
    stages = {suffix: [str(p.relative_to(out)) for p in sorted((out / "raw").rglob(f"*.{suffix}"))]
              for suffix in ("ttir", "ttgir", "llir", "ptx", "cubin")}
    return {"captures": captures, "backend_stages": stages}
