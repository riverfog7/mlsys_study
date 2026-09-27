"""Small CUDA workloads; shared inputs/modes for correctness and profiling."""
from __future__ import annotations

import copy

WORKLOADS = ("gelu", "layernorm", "residual_layernorm", "mlp", "vocab_projection", "block1")


def add_workload_args(parser):
    parser.add_argument("--workload", choices=WORKLOADS, default="gelu")
    parser.add_argument("--batch", type=int, default=1)
    parser.add_argument("--sequence", type=int, default=8)
    parser.add_argument("--channels", type=int, default=32)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--attention", choices=("manual", "sdpa"), default="manual")
    parser.add_argument("--training", action="store_true")


def build_workload(args):
    import torch
    from torch import nn
    from study02.examples import ManualGELU, ManualLayerNorm, ResidualLayerNorm
    from study02.model import GPT, GPTConfig, MLP
    if min(args.batch, args.sequence, args.channels, args.heads, args.vocab_size) < 1:
        raise ValueError("All shapes must be positive")
    dtype = getattr(torch, args.dtype)
    config = GPTConfig(block_size=args.sequence, vocab_size=args.vocab_size,
                       n_embd=args.channels, n_head=args.heads, attention_impl=args.attention)
    if args.workload == "block1":
        model = GPT(config).to(device="cuda", dtype=dtype)
        tokens = torch.randint(args.vocab_size, (args.batch, args.sequence), device="cuda")
        targets = torch.randint(args.vocab_size, tokens.shape, device="cuda")
        inputs = (tokens, targets) if args.training else (tokens,)
    else:
        constructors = {
            "gelu": lambda: ManualGELU(),
            "layernorm": lambda: ManualLayerNorm(args.channels),
            "residual_layernorm": lambda: ResidualLayerNorm(args.channels),
            "mlp": lambda: MLP(config),
            "vocab_projection": lambda: nn.Linear(args.channels, args.vocab_size, bias=False),
        }
        model = constructors[args.workload]().to(device="cuda", dtype=dtype)
        x = torch.randn(args.batch, args.sequence, args.channels, device="cuda", dtype=dtype,
                        requires_grad=args.training)
        inputs = (x, torch.randn_like(x)) if args.workload == "residual_layernorm" else (x,)
    model.train(args.training)
    return model, inputs


def output_tensor(output):
    return output[0] if isinstance(output, tuple) else output


def step(model, inputs, training):
    import torch
    model.zero_grad(set_to_none=True)
    for x in inputs:
        if x.is_floating_point() and x.grad is not None:
            x.grad = None
    with torch.set_grad_enabled(training):
        output = model(*inputs)
        if training:
            loss = output[1] if isinstance(output, tuple) and output[1] is not None else output_tensor(output).float().sum()
            loss.backward()
    return output


def clone_inputs(inputs):
    return tuple(x.detach().clone().requires_grad_(x.requires_grad) for x in inputs)


def compare(model, compiled, inputs, training, dtype):
    import torch
    reference = copy.deepcopy(model)
    reference_inputs = clone_inputs(inputs)
    expected = step(reference, reference_inputs, training)
    actual = step(compiled, inputs, training)
    rtol, atol = (1e-4, 1e-5) if dtype == "float32" else (2e-2, 2e-2)
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)
    gradient_count = 0
    if training:
        for (name, p), (other_name, ref) in zip(model.named_parameters(), reference.named_parameters()):
            assert name == other_name
            torch.testing.assert_close(p.grad, ref.grad, rtol=rtol, atol=atol)
            gradient_count += p.grad is not None
        for x, ref in zip(inputs, reference_inputs):
            torch.testing.assert_close(x.grad, ref.grad, rtol=rtol, atol=atol)
            gradient_count += x.grad is not None
    tensor = output_tensor(actual).detach()
    return {"passed": True, "rtol": rtol, "atol": atol, "gradients_checked": gradient_count,
            "output_shape": list(tensor.shape),
            "max_abs_error": (tensor - output_tensor(expected).detach()).abs().max().item()}
