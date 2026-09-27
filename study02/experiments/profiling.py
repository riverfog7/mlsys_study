"""Measure one execution mode in one fresh CUDA process, with actual CUPTI trace."""
from study02.runtime import experiment_main
from study02.workloads import add_workload_args


def configure(parser):
    add_workload_args(parser)
    parser.add_argument("--mode", choices=("eager", "compiled"), default="compiled")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)


def run(args, out):
    import json
    import statistics
    import time
    import torch
    from study02.workloads import build_workload, step, compare

    if args.warmup < 1 or args.iterations < 1 or args.repeats < 3:
        raise ValueError("warmup/iterations must be positive and repeats must be at least 3")
    model, inputs = build_workload(args)
    fn = torch.compile(model, fullgraph=True, dynamic=False) if args.mode == "compiled" else model
    # Set grad mode once around the benchmark. Inference does not need zero_grad
    # or parameter traversal on every call; that overhead hides tiny GPU work.
    def call():
        return step(fn, inputs, True) if args.training else fn(*inputs)

    event_samples, wall_samples = [], []
    profile_iterations = 3
    with torch.set_grad_enabled(args.training):
        torch.cuda.synchronize()
        start = time.perf_counter()
        call()
        torch.cuda.synchronize()
        first_ms = (time.perf_counter() - start) * 1000
        for _ in range(args.warmup):
            call()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeats):
            event_start = torch.cuda.Event(enable_timing=True)
            event_end = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            wall_start = time.perf_counter()
            event_start.record()
            for _ in range(args.iterations):
                call()
            event_end.record()
            event_end.synchronize()
            wall_samples.append((time.perf_counter() - wall_start) * 1000 / args.iterations)
            event_samples.append(event_start.elapsed_time(event_end) / args.iterations)
        memory = {"baseline_allocated_bytes": baseline,
                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                  "peak_reserved_bytes": torch.cuda.max_memory_reserved()}
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                torch.profiler.ProfilerActivity.CUDA],
                                    record_shapes=True, profile_memory=True) as prof:
            for _ in range(profile_iterations):
                with torch.profiler.record_function(f"study02::{args.workload}::{args.mode}"):
                    call()
                prof.step()
            torch.cuda.synchronize()
    event_ms, wall_ms = statistics.median(event_samples), statistics.median(wall_samples)
    q25, _, q75 = statistics.quantiles(event_samples, n=4, method="inclusive")
    if event_ms <= 0:
        raise RuntimeError("CUDA event interval is too short; increase iterations")
    trace_path = out / "raw" / "profiler_trace.json"
    prof.export_chrome_trace(str(trace_path))
    trace = json.loads(trace_path.read_text())
    events = [e for e in trace["traceEvents"] if e.get("ph") == "X" and e.get("dur", 0) > 0]
    kernels = [e for e in events if e.get("cat") == "kernel"]
    if not kernels:
        raise RuntimeError("CUDA profiler captured no kernels. Check CUPTI/driver; host events are not substituted.")
    selected = [e for e in events if e.get("cat") in {"kernel", "cpu_op", "cuda_runtime", "user_annotation"}]
    origin = min(e["ts"] for e in selected)
    timeline = [{"name": e["name"], "category": e.get("cat"),
                 "lane": f"GPU stream {e.get('args', {}).get('stream', e.get('tid'))}" if e.get("cat") == "kernel" else "CPU host",
                 "start_us": e["ts"] - origin, "duration_us": e["dur"]} for e in selected]
    correctness = compare(model, fn, inputs, args.training, args.dtype)
    return {"workload": args.workload, "attention": args.attention, "training": args.training,
            "mode": args.mode, "first_call_ms": first_ms, "steady_wall_ms": wall_ms,
            "steady_cuda_event_ms": event_ms, "warmup": args.warmup, "iterations": args.iterations,
            "repeats": args.repeats, "cuda_event_samples_ms": event_samples, "wall_samples_ms": wall_samples,
            "cuda_event_q25_ms": q25, "cuda_event_q75_ms": q75,
            "input_shape": list(inputs[0].shape),
            "input_elements": inputs[0].numel(),
            "throughput_million_elements_s": inputs[0].numel() / event_ms / 1000,
            "cuda_kernel_duration_ms_per_call": sum(e["dur"] for e in kernels) / profile_iterations / 1000,
            "cuda_launches_per_call": len(kernels) / profile_iterations,
            "memory": memory, "timeline": timeline, "cuda_kernel_count": len(kernels),
            "profile_iterations": profile_iterations,
            "kernel_names": sorted({e["name"] for e in kernels}),
            "trace_file": "raw/profiler_trace.json", "correctness": correctness}


if __name__ == "__main__":
    experiment_main("profiling", run, configure)
