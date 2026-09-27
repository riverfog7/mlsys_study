"""Serialize actual FX graphs without executing or reconstructing their source."""
from __future__ import annotations

from study02.runtime import write_json


def tensor_metadata(value):
    import torch
    if not isinstance(value, torch.Tensor):
        return {"python_type": type(value).__name__, "symbol": str(value)}
    def dim(n):
        return n if isinstance(n, int) else str(n)
    return {
        "python_type": type(value).__name__, "shape": [dim(n) for n in value.shape],
        "stride": [dim(n) for n in value.stride()], "dtype": str(value.dtype),
        "device": str(value.device), "requires_grad": value.requires_grad,
    }


def graph_data(graph_or_module, stage):
    import torch
    graph = graph_or_module.graph if isinstance(graph_or_module, torch.fx.GraphModule) else graph_or_module
    nodes, edges = [], []
    for node in graph.nodes:
        val = node.meta.get("val", node.meta.get("example_value"))
        metadata = tensor_metadata(val) if val is not None else {}
        nodes.append({
            "id": node.name, "op": node.op, "target": str(node.target),
            "args": str(node.args), "kwargs": str(node.kwargs), "metadata": metadata,
            "source": str(node.meta.get("stack_trace", ""))[-1400:],
            "module": str(node.meta.get("nn_module_stack", "")),
        })
        edges.extend({"source": parent.name, "target": node.name} for parent in node.all_input_nodes)
    output = next(node for node in graph.nodes if node.op == "output")
    returns = []
    torch.fx.map_arg(output.args, lambda node: returns.append(node.name))
    return {"stage": stage, "provenance": "current execution", "nodes": nodes, "edges": edges,
            "returns": returns}


def save_graph(graph_or_module, out, stage):
    data = graph_data(graph_or_module, stage)
    write_json(out / "raw" / f"{stage}.json", data)
    module = getattr(graph_or_module, "owning_module", graph_or_module)
    if hasattr(module, "code"):
        (out / "raw" / f"{stage}.py").write_text(module.code, encoding="utf-8")
    return data
