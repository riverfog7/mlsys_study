"""Capture CUDA FX graphs, guards, symbolic shapes and Python specializations."""
from study02.runtime import experiment_main


def run(args, out):
    import torch
    from study02.examples import ManualGELU, tiny, two_regions, data_dependent, python_branch
    from study02.graphs import save_graph

    dtype = getattr(torch, args.dtype)
    graphs = []
    def capture(gm, inputs):
        graphs.append(save_graph(gm, out, f"dynamo_{len(graphs)}"))
        return gm.forward

    x = torch.tensor([-2., -1., 0., 1., 2.], device="cuda", dtype=dtype)
    compiled = torch.compile(ManualGELU(), backend=capture, fullgraph=True, dynamic=False)
    output = compiled(x)
    torch.testing.assert_close(output, ManualGELU()(x))
    gelu_graph = graphs[0]
    explanation = torch._dynamo.explain(ManualGELU())(x)
    guards = [{"name": g.name, "source": str(g.source), "types": g.guard_types,
               "conditions": g.code_list} for g in explanation.out_guards]
    shape_runs = []
    for dynamic in (False, True):
        torch._dynamo.reset()
        counts = []
        def counter(gm, inputs):
            graphs.append(save_graph(gm, out, f"dynamic_{dynamic}_{len(graphs)}"))
            counts.append(1)
            return gm.forward
        fn = torch.compile(tiny, backend=counter, fullgraph=True, dynamic=dynamic)
        w = torch.randn(4, 3, device="cuda", dtype=dtype)
        for index, batch in enumerate([2, 2, 5, 7]):
            values = torch.randn(batch, 4, device="cuda", dtype=dtype)
            torch.testing.assert_close(fn(values, w), tiny(values, w))
            shape_runs.append({"dynamic": dynamic, "call": index + 1, "batch": batch,
                               "captures_so_far": len(counts)})
    torch._dynamo.reset()
    broken = torch._dynamo.explain(two_regions)(x)
    break_graphs = [save_graph(gm, out, f"break_region_{i}") for i, gm in enumerate(broken.graphs)]
    expected_error = None
    torch._dynamo.reset()
    try:
        torch.compile(two_regions, backend="eager", fullgraph=True)(x)
    except torch._dynamo.exc.Unsupported as exc:
        expected_error = str(exc)
    if expected_error is None:
        raise AssertionError("fullgraph=True should reject the deliberate graph break")
    dependent = torch._dynamo.explain(data_dependent)(x + 1)
    torch._dynamo.reset()
    branch_graphs = []
    def branch_backend(gm, inputs):
        branch_graphs.append(save_graph(gm, out, f"branch_{len(branch_graphs)}"))
        return gm.forward
    branch = torch.compile(python_branch, backend=branch_backend, fullgraph=True)
    branch(x, None)
    branch(x, 1.0)
    # A Python loop over a fixed ModuleList is specialized/unrolled.
    loop_model = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Sigmoid()).cuda()
    loop_graphs = []
    def loop_backend(gm, inputs):
        loop_graphs.append(save_graph(gm, out, "module_loop"))
        return gm.forward
    torch.compile(loop_model, backend=loop_backend, fullgraph=True)(x)
    return {"gelu_graph": gelu_graph, "graphs": graphs, "guards": guards,
            "input": x.tolist(), "output": output.tolist(), "shape_runs": shape_runs,
            "break_graphs": break_graphs, "graph_count": broken.graph_count,
            "break_count": broken.graph_break_count,
            "break_reasons": [str(r.reason) for r in broken.break_reasons],
            "fullgraph_expected_error": expected_error,
            "data_dependent": {"graphs": dependent.graph_count,
                               "reasons": [str(r.reason) for r in dependent.break_reasons]},
            "branch_graphs": branch_graphs, "loop_graph": loop_graphs[0]}


if __name__ == "__main__":
    experiment_main("dynamo", run)
