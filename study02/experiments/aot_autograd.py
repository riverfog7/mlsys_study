"""Generate joint, forward and backward graphs and preserve mutation semantics."""
from study02.runtime import experiment_main


def run(args, out):
    import torch
    from functorch.compile import make_boxed_func
    from torch._dynamo.backends.common import aot_autograd
    from torch._functorch.partitioners import default_partition
    from torch.fx.experimental.proxy_tensor import make_fx
    from study02.examples import ManualGELU, mutate_view
    from study02.graphs import save_graph, tensor_metadata

    graphs, runtime_values = {}, {}
    def partition(gm, inputs, **kwargs):
        graphs["joint"] = save_graph(gm, out, "aot_joint")
        return default_partition(gm, inputs, **kwargs)

    def compiler(stage):
        def capture(gm, example_inputs):
            graphs[stage] = save_graph(gm, out, f"aot_{stage}")
            def execute(*inputs):
                outputs = gm(*inputs)
                runtime_values[stage] = {
                    "inputs": [{**tensor_metadata(x), "values": x.detach().tolist()} for x in inputs],
                    "outputs": [{**tensor_metadata(x), "values": x.detach().tolist()} for x in outputs],
                }
                return outputs
            return make_boxed_func(execute)
        return capture

    dtype = getattr(torch, args.dtype)
    x = torch.tensor([-2., -1., 0., 1., 2.], device="cuda", dtype=dtype, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    expected = ManualGELU()(reference_x)
    expected.sum().backward()
    backend = aot_autograd(fw_compiler=compiler("forward"), bw_compiler=compiler("backward"),
                          partition_fn=partition)
    compiled = torch.compile(ManualGELU(), backend=backend, fullgraph=True, dynamic=False)
    output = compiled(x)
    output.sum().backward()
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(x.grad, reference_x.grad)
    if set(graphs) != {"joint", "forward", "backward"}:
        raise AssertionError(f"Missing AOT graph: {set(graphs)}")

    original = torch.zeros(2, 2, device="cuda", dtype=dtype)
    functional_input = torch.zeros_like(original)
    original_result = mutate_view(original)
    functional = torch.func.functionalize(mutate_view)
    functional_result = functional(functional_input)
    torch.testing.assert_close(original_result, functional_result)
    torch.testing.assert_close(original, functional_input)
    mutation_graph = save_graph(make_fx(functional)(torch.zeros_like(original)), out, "functionalized")
    fw_returns = graphs["forward"]["returns"]
    bw_inputs = [n["id"] for n in graphs["backward"]["nodes"] if n["op"] == "placeholder"]
    return {"graphs": graphs, "runtime_values": runtime_values,
            "interface": {"user_output": fw_returns[0], "saved_values": fw_returns[1:],
                          "backward_inputs": bw_inputs},
            "input": x.detach().tolist(), "output": output.detach().tolist(),
            "gradient": x.grad.tolist(), "reference_gradient": reference_x.grad.tolist(),
            "functionalized_graph": mutation_graph,
            "mutation": {"before": torch.zeros_like(original).tolist(), "eager_input_after": original.tolist(),
                         "functional_input_after": functional_input.tolist(),
                         "external_mutation_preserved": True}, "correctness": {"passed": True}}


if __name__ == "__main__":
    experiment_main("aot_autograd", run)
