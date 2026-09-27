"""Compile a CUDA workload and collect actual FX, scheduler IR and Triton code."""
from study02.runtime import experiment_main
from study02.workloads import add_workload_args


def run(args, out):
    import torch
    from torch._inductor import config
    from study02.graphs import save_graph
    from study02.workloads import build_workload, compare
    from study02.artifacts import collect_inductor

    model, inputs = build_workload(args)
    graphs = {"input": [], "transformed": []}
    def capture(stage):
        def hook(graph):
            graphs[stage].append(save_graph(graph, out, f"inductor_{stage}_{len(graphs[stage])}"))
        return hook
    with config.patch({"post_grad_custom_pre_pass": capture("input"),
                       "post_grad_custom_post_pass": capture("transformed"),
                       "fx_graph_cache": False}):
        compiled = torch.compile(model, backend="inductor", fullgraph=True, dynamic=False)
        correctness = compare(model, compiled, inputs, args.training, args.dtype)
    torch.cuda.synchronize()
    artifacts = collect_inductor(out)
    return {"workload": args.workload, "training": args.training, "attention": args.attention,
            "graphs": graphs, "correctness": correctness, **artifacts}


if __name__ == "__main__":
    experiment_main("inductor", run, add_workload_args)
