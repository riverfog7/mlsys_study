"""GPT-2 profiling support; plots and interpretation live in main.ipynb."""

import contextlib
from collections import Counter
import copy
import csv
from datetime import datetime, timezone
import io
from importlib import metadata as package_metadata
import json
from pathlib import Path
import re
import runpy
import statistics
import subprocess
import sys
import tempfile
import time
from types import MethodType
import warnings

import torch


def _forward(model, ids):
    return model(input_ids=ids, use_cache=False, return_dict=True).logits


def _captured_forward_events(events):
    """Exclude warmups by CPU scope and launch IDs, not drifting GPU timestamps."""
    markers = [event for event in events if event.get("cat") == "user_annotation"
               and event.get("name") == "study03/captured_forward"]
    if len(markers) != 1:
        raise RuntimeError("The profiler trace is missing its unique captured-forward scope.")
    marker = markers[0]
    cpu = [event for event in events
           if event.get("cat") in ("cpu_op", "user_annotation", "cuda_runtime")
           and event.get("tid") == marker["tid"]
           and marker["ts"] <= event.get("ts", 0) < marker["ts"] + marker["dur"]]
    external_ids = {event["args"]["External id"] for event in cpu
                    if event.get("args", {}).get("External id") is not None}
    launches = {event["args"]["correlation"] for event in cpu
                if event.get("cat") == "cuda_runtime"
                and event.get("name", "").startswith(("cudaLaunchKernel", "cudaLaunchCooperativeKernel"))
                and event.get("args", {}).get("correlation") is not None}
    # Direct CUDA extensions need launch correlations: their kernel records can
    # lack an ATen External id even though their cudaLaunchKernel is recorded.
    kernels = [event for event in events if event.get("cat") == "kernel"
               and (event.get("args", {}).get("External id") in external_ids
                    or event.get("args", {}).get("correlation") in launches)]
    recorded = {event.get("args", {}).get("correlation") for event in kernels}
    return cpu + kernels, launches - recorded


def _capture_cuda_trace(run, *, record_shapes=False):
    """Warm CUPTI during active collection, then retain one complete forward.

    In long-lived notebook kernels, initial GPU records can fall outside
    Kineto's capture window (pytorch/pytorch#192021). Warmup before starting the
    profiler does not fix that. These extra runs are outside latency/memory
    measurements, and their events are excluded from the returned trace.
    """
    for warmup_seconds in (0.05, 0.2):
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=record_shapes,
        ) as profiler:
            deadline = time.perf_counter() + warmup_seconds
            while True:
                warmup_output = run()
                torch.cuda.synchronize()
                del warmup_output
                if time.perf_counter() >= deadline:
                    break
            with torch.profiler.record_function("study03/captured_forward"):
                output = run()
                torch.cuda.synchronize()
        with tempfile.TemporaryDirectory(prefix="study03-trace-") as directory:
            path = Path(directory) / "trace.json"
            profiler.export_chrome_trace(str(path))
            events = json.loads(path.read_text())["traceEvents"]
        events, missing_launches = _captured_forward_events(events)
        if any(event.get("cat") == "kernel" for event in events) and not missing_launches:
            return output, events
        del output
    raise RuntimeError("CUDA profiler did not capture a complete forward after in-profiler warmup. "
                       f"Missing kernel launch records: {len(missing_launches)}. "
                       "CUDA kernel verification could not complete.")


@contextlib.contextmanager
def _module_ranges(model):
    handles, opened = [], []

    def tag(module, name):
        def enter(_module, _args):
            scope = torch.profiler.record_function(name)
            scope.__enter__()
            opened.append(scope)

        def leave(_module, _args, _result):
            opened.pop().__exit__(None, None, None)

        handles.extend([module.register_forward_pre_hook(enter),
                        module.register_forward_hook(leave)])

    for i, block in enumerate(model.transformer.h):
        tag(block.attn, f"study03/block{i:02d}/attention")
        tag(block.attn.c_attn, f"study03/block{i:02d}/qkv")
        tag(block.attn.c_proj, f"study03/block{i:02d}/out")
        tag(block.mlp, f"study03/block{i:02d}/mlp")
    tag(model.lm_head, "study03/lm_head")
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        while opened:
            opened.pop().__exit__(None, None, None)


def _parse_trace(events):
    ops = {e["args"]["External id"]: e for e in events
           if e.get("cat") == "cpu_op" and "External id" in e.get("args", {})}
    scopes = [e for e in events if e.get("name", "").startswith("study03/")
              and e.get("ph") == "X"]
    kernels = sorted((e for e in events if e.get("cat") == "kernel"),
                     key=lambda e: e["ts"])
    assert kernels, "No CUDA kernel events were recorded."
    rows = []
    bmm_index, bmm_stage = {}, {}
    for kernel in kernels:
        op = ops.get(kernel.get("args", {}).get("External id"))
        assert op is not None, f"Unmapped CUDA kernel: {kernel['name']}"
        parents = [e for e in scopes if e["tid"] == op["tid"]
                   and e["ts"] <= op["ts"] < e["ts"] + e["dur"]]
        scope = min(parents, key=lambda e: e["dur"])["name"] if parents else ""
        block = int(scope.split("/")[1][5:]) if "/block" in scope else None
        operator = op["name"]
        stage = "Other / norms"
        if scope.endswith("/qkv"):
            stage = "QKV projection"
        elif scope.endswith("/out"):
            stage = "Output projection"
        elif scope.endswith("/mlp"):
            stage = "MLP"
        elif scope.endswith("/lm_head"):
            stage = "LM head"
        elif scope.endswith("/attention"):
            if operator == "aten::bmm":
                ext = op["args"]["External id"]
                if ext not in bmm_stage:
                    index = bmm_index.get(block, 0)
                    assert index < 2, "Expected two eager-attention batched matmuls."
                    bmm_stage[ext] = ("QKᵀ", "PV")[index]
                    bmm_index[block] = index + 1
                stage = bmm_stage[ext]
            elif "softmax" in operator:
                stage = "Softmax"
            elif operator in ("aten::div", "aten::where", "aten::fill_", "aten::add"):
                stage = "Scale / causal mask"
            else:
                stage = "Attention copies"
        args = kernel["args"]
        rows.append(dict(name=kernel["name"], operator=operator, stage=stage,
                         block=block, scope=scope,
                         start_us=kernel["ts"] - kernels[0]["ts"],
                         duration_us=kernel["dur"],
                         grid=args.get("grid"), threads=args.get("block"),
                         stream=args.get("stream")))
    assert all(n == 2 for n in bmm_index.values())
    return rows


def profile_forward(model, ids, warmup=5, repeats=20):
    """Time a normal forward, then trace one separate, instrumented forward."""
    assert ids.is_cuda and next(model.parameters()).is_cuda
    assert not model.training and model.config._attn_implementation == "eager"
    assert 1 < ids.shape[1] <= model.config.n_positions
    with torch.inference_mode():
        for _ in range(warmup):
            _forward(model, ids)
        torch.cuda.synchronize()
        elapsed = []
        for _ in range(repeats):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            logits = _forward(model, ids)
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end))
        reference = logits[0, -1, :16].cpu().tolist()
        del logits
        with _module_ranges(model):
            logits, events = _capture_cuda_trace(lambda: _forward(model, ids), record_shapes=True)
        torch.testing.assert_close(logits[0, -1, :16].cpu().float(),
                                   torch.tensor(reference), rtol=1e-5, atol=1e-5)
        del logits
    kernels = _parse_trace(events)
    assert sum(r["stage"] == "QKᵀ" for r in kernels) >= model.config.n_layer
    assert sum(r["stage"] == "PV" for r in kernels) >= model.config.n_layer
    return dict(tokens=ids.shape[1], elapsed_ms=elapsed,
                median_ms=statistics.median(elapsed), kernels=kernels,
                logits_sample=reference)


METRICS = ["gpu__time_duration.sum", "dram__bytes_read.sum", "dram__bytes_write.sum",
           "dram__throughput.avg.pct_of_peak_sustained_elapsed",
           "sm__throughput.avg.pct_of_peak_sustained_elapsed",
           "lts__t_sector_hit_rate.pct"]


def profile_counters(model_id, revision, dtype, ids, trace):
    """Run the same full GPT-2 forward; collect hardware counters only in block 0 attention."""
    config = dict(model_id=model_id, revision=revision, dtype=str(dtype).split(".")[-1],
                  ids=ids.cpu().tolist())
    command = ["sudo", "-n", "/usr/bin/ncu", "--profile-from-start", "off",
               "--cache-control", "none", "--clock-control", "none",
               "--metrics", ",".join(METRICS), "--csv",
               sys.executable, "-B", str(Path(__file__).resolve()), "--counters"]
    completed = subprocess.run(command, input=json.dumps(config), capture_output=True,
                               text=True, timeout=240, cwd=Path(__file__).resolve().parents[1])
    output = completed.stdout + "\n" + completed.stderr
    if completed.returncode or "==ERROR==" in output:
        raise RuntimeError("Nsight Compute collection failed:\n" + output[-7000:])
    marker = "STUDY03_LOGITS="
    samples = [json.loads(line[len(marker):]) for line in output.splitlines() if line.startswith(marker)]
    assert len(samples) == 1, "Counter run did not return its forward output check."
    torch.testing.assert_close(torch.tensor(samples[0]), torch.tensor(trace["logits_sample"]),
                               rtol=1e-5, atol=1e-5)
    lines = output.splitlines()
    header = next((i for i, line in enumerate(lines) if line.startswith('"ID","Process ID"')), None)
    if header is None:
        raise RuntimeError("No counter rows:\n" + output[-7000:])
    csv_text = "\n".join(line for line in lines[header:] if line.startswith('"'))
    by_id = {}
    for row in csv.DictReader(io.StringIO(csv_text)):
        record = by_id.setdefault(int(row["ID"]), dict(name=row["Kernel Name"], metrics={},
                                                        grid=row["Grid Size"], threads=row["Block Size"]))
        record["metrics"][row["Metric Name"]] = float(row["Metric Value"].replace(",", ""))
    expected = [k for k in trace["kernels"] if k["block"] == 0 and k["scope"].split("/")[-1] in ("attention", "qkv", "out")]
    counters = [by_id[k] for k in sorted(by_id)]
    assert len(counters) == len(expected), f"Kernel counts differ: ncu={len(counters)}, trace={len(expected)}"
    for counter, kernel in zip(counters, expected):
        for field in ("grid", "threads"):
            shape = [int(n) for n in re.findall(r"\d+", counter[field])]
            assert shape == kernel[field], f"Kernel launch mismatch: {counter['name']}"
        def base(name):
            return re.sub(r"\s+", "", name.replace("<unnamed>", "(anonymous namespace)")).split("<")[0]
        assert base(counter["name"]) == base(kernel["name"]), f"Kernel name mismatch: {counter['name']} / {kernel['name']}"
        assert set(METRICS) <= counter["metrics"].keys()
        counter.update(stage=kernel["stage"], operator=kernel["operator"],
                       trace_duration_us=kernel["duration_us"])
    return dict(kernels=counters, command=command,
                notes=[line for line in lines if "==WARNING==" in line],
                tokens=trace["tokens"])


def _counter_child():
    config = json.load(sys.stdin)
    common = runpy.run_path(str(Path(__file__).resolve().parents[1] / "init_project.py"))
    from transformers import AutoModelForCausalLM
    assert common["DEVICE"].type == "cuda"
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"], revision=config["revision"], dtype=getattr(torch, config["dtype"]),
        attn_implementation="eager", use_safetensors=True, local_files_only=True,
        cache_dir=common["HF_HOME"] / "hub",
    ).to(common["DEVICE"]).eval()
    ids = torch.tensor(config["ids"], dtype=torch.long, device=common["DEVICE"])
    with torch.inference_mode():
        for _ in range(5):
            _forward(model, ids)
        torch.cuda.synchronize()

        def start(_module, _args):
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStart()

        def stop(_module, _args, _result):
            torch.cuda.synchronize()
            torch.cuda.cudart().cudaProfilerStop()

        attention = model.transformer.h[0].attn
        handles = [attention.register_forward_pre_hook(start), attention.register_forward_hook(stop)]
        try:
            logits = _forward(model, ids)
            torch.cuda.synchronize()
            print("STUDY03_LOGITS=" + json.dumps(logits[0, -1, :16].cpu().tolist()), flush=True)
        finally:
            for handle in handles:
                handle.remove()


def _comparison_error(reference, actual):
    """Compare every output value, limiting temporary storage for full logits."""
    assert reference.shape == actual.shape
    maximum, squared_error, squared_reference, count = 0.0, 0.0, 0.0, 0
    for expected, observed in zip(reference.flatten().split(1_000_000),
                                  actual.flatten().split(1_000_000)):
        expected, observed = expected.float(), observed.float()
        difference = observed - expected
        maximum = max(maximum, difference.abs().max().item())
        squared_error += difference.square().sum().item()
        squared_reference += expected.square().sum().item()
        count += expected.numel()
    return dict(max_abs=maximum, rmse=(squared_error / count) ** 0.5,
                relative_l2=(squared_error / max(squared_reference, 1e-30)) ** 0.5,
                values_compared=count)


def _comparison_profile(run, expected_operator=None, expected_kernel=None):
    """The trace is a separate run: its overhead never enters latency samples."""
    output, events = _capture_cuda_trace(run)
    del output
    operators = dict(Counter(event["name"] for event in events if event.get("cat") == "cpu_op"))
    if expected_operator and expected_operator not in operators:
        raise AssertionError(f"Required backend was not observed: {expected_operator}")
    if expected_operator and "aten::_scaled_dot_product_attention_math" in operators:
        raise AssertionError("An SDPA math fallback appeared in a forced-backend run.")
    kernels = {}
    for event in events:
        if event.get("cat") == "kernel":
            item = kernels.setdefault(event["name"], dict(name=event["name"], count=0,
                                                           total_us=0.0))
            item["count"] += 1
            item["total_us"] += event["dur"]
    if not kernels:
        raise AssertionError("No CUDA kernels recorded in the comparison trace.")
    if expected_kernel:
        if not any(expected_kernel in name for name in kernels):
            raise AssertionError(f"Required CUDA kernel was not observed: {expected_kernel}")
        if any("scaled_dot_product" in name for name in operators):
            raise AssertionError("The Turing extension unexpectedly called an SDPA backend.")
    return dict(operators=operators, kernels=list(kernels.values()),
                kernel_count=sum(item["count"] for item in kernels.values()))


def _prediction_error(reference, actual):
    """Show whether FP16 logit differences also change next-token predictions."""
    agreement, variation, maximum_probability, maximum_variation = 0, 0.0, 0.0, 0.0
    positions = reference.shape[0] * reference.shape[1]
    for expected, observed in zip(reference.split(32, dim=1), actual.split(32, dim=1)):
        expected, observed = expected.float(), observed.float()
        agreement += (expected.argmax(-1) == observed.argmax(-1)).sum().item()
        difference = (expected.softmax(-1) - observed.softmax(-1)).abs()
        total_variation = difference.sum(-1) * 0.5
        maximum_probability = max(maximum_probability, difference.max().item())
        maximum_variation = max(maximum_variation, total_variation.max().item())
        variation += total_variation.sum().item()
    return dict(argmax_agreement=agreement / positions,
                max_probability_difference=maximum_probability,
                mean_total_variation=variation / positions,
                max_total_variation=maximum_variation)


def _turing_core(extension, query, key, value):
    """The upstream kernel ignores strides: provide fully packed BSHD tensors."""
    if query.shape[-1] not in (64, 96, 128) or query.dtype != torch.float16:
        raise ValueError("FlashAttention Turing requires FP16 and head dimension 64, 96, or 128.")
    query, key, value = [tensor.transpose(1, 2).contiguous() for tensor in (query, key, value)]
    return extension.fwd(query, key, value, query.shape[-1] ** -0.5, True)[0]


def _comparison_forward(backend, extension=None):
    """Use identical GPT-2 projections, mask preparation, and layout handling."""
    from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward

    def forward(module, hidden_states, past_key_values=None, cache_position=None,
                attention_mask=None, head_mask=None, encoder_hidden_states=None,
                encoder_attention_mask=None, output_attentions=False, **kwargs):
        if (past_key_values is not None or attention_mask is not None or head_mask is not None
                or encoder_hidden_states is not None or encoder_attention_mask is not None
                or output_attentions or module.training):
            raise ValueError("The comparison supports only unpadded causal prefill in eval mode.")
        projected = module.c_attn(hidden_states).split(module.split_size, dim=2)
        query, key, value = [tensor.view(*tensor.shape[:-1], module.num_heads, module.head_dim)
                             .transpose(1, 2) for tensor in projected]
        if backend == "eager":
            output = eager_attention_forward(module, query, key, value, None)[0]
        elif backend == "turing":
            output = _turing_core(extension, query, key, value)
        else:
            output = torch.nn.functional.scaled_dot_product_attention(
                query, key, value, dropout_p=0.0, is_causal=True).transpose(1, 2)
        output = output.reshape(*output.shape[:-2], -1).contiguous()
        return module.resid_dropout(module.c_proj(output)), None
    return forward


def compare_attention(model, ids, lengths=(128, 512, 1024), warmup=5, repeats=20,
                      turing_revision=None):
    """Compare causal GPT-2 eager, forced SDPA, and optional Turing Flash in FP16.

    A private FP16 model copy preserves the caller's weights, dtype, and config.
    Both paths use the same copy, inputs, and batch size, without a KV cache.
    Core timing excludes QKV/output projections; model timing includes all logits.
    Unsupported Flash is reported explicitly; efficient attention is a separate
    backend, never a replacement labeled Flash. No profiler runs are timed.
    The optional ``flash_attn_turing`` extension is the third-party ssiu port,
    not an official FA1 kernel. Supply its verified build commit as turing_revision.
    Layout copies required by that extension are included in timed calls.
    Every full-model backend uses the same private attention adapter: identical
    QKV/output projections, no redundant external causal mask, and identical
    output-layout handling. Only the attention core implementation changes.

    Returns JSON-serializable metadata, backend statuses, and one record per length.
    Each record has ``attention_core`` and ``model_forward`` backend dictionaries.
    ``peak_extra_bytes`` is allocator peak minus resident baseline, not DRAM I/O.
    """
    from torch.nn.attention import SDPBackend, sdpa_kernel
    from transformers import __version__ as transformers_version
    from transformers.models.gpt2.modeling_gpt2 import eager_attention_forward

    lengths = tuple(int(length) for length in lengths)
    if not lengths or min(lengths) < 2 or max(lengths) > model.config.n_positions:
        raise ValueError("Lengths must lie between 2 and GPT-2's position limit.")
    if (ids.ndim != 2 or ids.shape[0] != 1 or ids.shape[1] < max(lengths)
            or not ids.is_cuda or next(model.parameters()).device != ids.device):
        raise ValueError("Supply a CUDA GPT-2 and batch-one CUDA IDs covering every length.")
    if warmup < 1 or repeats < 2:
        raise ValueError("Use at least one warmup and two measured repetitions.")
    if model.config.model_type != "gpt2":
        raise ValueError("This comparison extracts real QKV from GPT-2 block 0.")
    labels = {"eager": "Eager", "flash": "PyTorch SDPA FlashAttention",
              "efficient": "SDPA efficient (CUTLASS)", "turing": "FlashAttention Turing (ssiu)"}
    choices = {"flash": SDPBackend.FLASH_ATTENTION,
               "efficient": SDPBackend.EFFICIENT_ATTENTION}
    expected = {"flash": "aten::_scaled_dot_product_flash_attention",
                "efficient": "aten::_scaled_dot_product_efficient_attention"}
    extension, turing_unavailable = None, None
    try:
        import flash_attn_turing as extension
    except (ImportError, OSError) as error:
        turing_unavailable = dict(status="unavailable", reason=f"flash_attn_turing import failed: {error}")
    try:
        turing_version = package_metadata.version("flash-attn-turing")
    except package_metadata.PackageNotFoundError:
        turing_version = None
    turing_head_dimensions = (64,) if "hdim64" in (turing_version or "") else (64, 96, 128)
    turing_build_variant = ("head-64 causal forward only" if "hdim64causalfwd" in (turing_version or "")
                            else "head-64 forward only" if "hdim64fwd" in (turing_version or "")
                            else "package default")
    result = dict(metadata=dict(
        measured_at_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        gpu=torch.cuda.get_device_name(ids.device),
        compute_capability=list(torch.cuda.get_device_capability(ids.device)),
        torch=torch.__version__, transformers=transformers_version,
        cuda=torch.version.cuda, model=getattr(model.config, "_name_or_path", "GPT-2"),
        model_revision=getattr(model.config, "_commit_hash", None),
        dtype="float16", batch=1, causal=True, use_cache=False, training=False,
        warmup=warmup, repeats=repeats,
        timing="CUDA events on default stream; backend order reversed on alternating repetitions",
        memory="peak memory_allocated minus baseline; includes returned output; not DRAM traffic",
        correctness="Approximate FP16 outputs; relative-L2 check is not identical predictions",
        model_adapter="Common private GPT-2 adapter for every backend; same projections/layout handling; causal mask handled inside each core",
        flash="Forced PyTorch Flash backend; no separate FA1 versus FA2 kernel comparison",
        turing=dict(source_url="https://github.com/ssiu/flash-attention-turing",
                    source_revision=turing_revision, package_version=turing_version,
                    build_variant=turing_build_variant,
                    supported_head_dimensions=list(turing_head_dimensions),
                    extension_path=getattr(extension, "__file__", None),
                    implementation="Third-party FA2-style Turing port; not official FA1",
                    layout="Packed BSHD copies included; upstream kernel uses default CUDA stream")),
        backends={name: dict(label=label, status="pending") for name, label in labels.items()},
        lengths=[])
    benchmark_model = copy.deepcopy(model).to(dtype=torch.float16).eval()
    attention = benchmark_model.transformer.h[0].attn
    if (not attention.scale_attn_weights or attention.scale_attn_by_inverse_layer_idx
            or attention.reorder_and_upcast_attn):
        raise ValueError("This comparison expects the pretrained GPT-2 attention scaling defaults.")
    if extension is not None and (attention.head_dim not in turing_head_dimensions
                                  or torch.cuda.get_device_capability(ids.device) != (7, 5)):
        turing_unavailable = dict(status="unsupported", reason=(
            f"This Turing build targets sm75 and head dimensions {list(turing_head_dimensions)}."))
    if turing_unavailable:
        result["backends"]["turing"].update(turing_unavailable)

    @contextlib.contextmanager
    def select(backend):
        original_implementation = benchmark_model.config._attn_implementation
        originals = []
        # All backends use the same custom mask early-exit. Eager applies its own
        # causal mask; fused cores receive causal=True. No global registry changes.
        benchmark_model.config._attn_implementation = "study03_comparison"
        forward = _comparison_forward(backend, extension)
        for block in benchmark_model.transformer.h:
            originals.append((block.attn, block.attn.forward))
            block.attn.forward = MethodType(forward, block.attn)
        try:
            with sdpa_kernel(choices[backend]) if backend in choices else contextlib.nullcontext():
                yield
        finally:
            for module, original in originals:
                module.forward = original
            benchmark_model.config._attn_implementation = original_implementation

    try:
        # The extension launches on the default stream; honor that for inputs, copies, and events.
        torch.cuda.synchronize(ids.device)
        with torch.cuda.device(ids.device), torch.cuda.stream(torch.cuda.default_stream(ids.device)), torch.inference_mode():
            for length in lengths:
                current_ids = ids[:, :length]
                positions = torch.arange(length, device=ids.device)[None, :]
                hidden = benchmark_model.transformer.wte(current_ids) + benchmark_model.transformer.wpe(positions)
                hidden = benchmark_model.transformer.h[0].ln_1(hidden)
                projected = attention.c_attn(hidden).split(attention.split_size, dim=2)
                query, key, value = [item.view(1, length, attention.num_heads, attention.head_dim)
                                     .transpose(1, 2) for item in projected]
                del hidden, projected

                def core(backend):
                    if backend == "eager":
                        return eager_attention_forward(attention, query, key, value, None)[0]
                    if backend == "turing":
                        return _turing_core(extension, query, key, value)
                    return torch.nn.functional.scaled_dot_product_attention(
                        query, key, value, dropout_p=0.0, is_causal=True).transpose(1, 2)

                record = dict(tokens=length, qkv_shape=list(query.shape),
                              attention_core={}, model_forward={})
                available = []
                for backend in labels:
                    if backend == "turing" and turing_unavailable:
                        record["attention_core"][backend] = turing_unavailable.copy()
                        record["model_forward"][backend] = turing_unavailable.copy()
                        continue
                    with warnings.catch_warnings(record=True) as caught:
                        warnings.simplefilter("always")
                        try:
                            with select(backend):
                                trial = core(backend)
                                del trial
                                trial = _forward(benchmark_model, current_ids)
                                del trial
                                torch.cuda.synchronize()
                        except RuntimeError as error:
                            unsupported = any(message in str(error).lower() for message in
                                              ("no available kernel", "no viable backend"))
                            if backend == "eager" or not unsupported:
                                raise
                            reason = " | ".join([str(error)] + [str(item.message) for item in caught])
                            status = dict(status="unsupported", reason=reason)
                            record["attention_core"][backend] = status.copy()
                            record["model_forward"][backend] = status.copy()
                            result["backends"][backend].update(status)
                            continue
                    available.append(backend)
                    result["backends"][backend]["status"] = "measured"

                for scope in ("attention_core", "model_forward"):
                    run = core if scope == "attention_core" else lambda backend: _forward(benchmark_model, current_ids)
                    for backend in available:
                        with select(backend):
                            for _ in range(warmup):
                                output = run(backend)
                                del output
                    torch.cuda.synchronize()
                    samples = {backend: [] for backend in available}
                    for repeat in range(repeats):
                        order = available if repeat % 2 == 0 else list(reversed(available))
                        for backend in order:
                            with select(backend):
                                start, stop = (torch.cuda.Event(enable_timing=True) for _ in range(2))
                                start.record()
                                output = run(backend)
                                stop.record()
                                stop.synchronize()
                                samples[backend].append(start.elapsed_time(stop))
                                del output
                    for backend in available:
                        with select(backend):
                            torch.cuda.synchronize()
                            baseline = torch.cuda.memory_allocated()
                            torch.cuda.reset_peak_memory_stats()
                            output = run(backend)
                            torch.cuda.synchronize()
                            peak = torch.cuda.max_memory_allocated()
                            del output
                            profile = _comparison_profile(
                                lambda: run(backend), expected.get(backend),
                                "flash_fwd_kernel" if backend == "turing" else None)
                        times = samples[backend]
                        record[scope][backend] = dict(
                            status="measured", elapsed_ms=times,
                            median_ms=statistics.median(times), min_ms=min(times), max_ms=max(times),
                            baseline_bytes=baseline, peak_allocated_bytes=peak,
                            peak_extra_bytes=peak - baseline, profile=profile)

                    # Accuracy runs happen after latency/memory collection and compare all values.
                    with select("eager"):
                        if scope == "attention_core":
                            reference = eager_attention_forward(
                                attention, query.float(), key.float(), value.float(), None)[0]
                            reference_name, tolerance = "FP32 eager on the same FP16 QKV", 0.003
                        else:
                            reference = run("eager")
                            reference_name, tolerance = "FP16 eager full logits", 0.005
                    for backend in available:
                        with select(backend):
                            output = run(backend)
                        error = _comparison_error(reference, output)
                        if scope == "model_forward":
                            error.update(_prediction_error(reference, output))
                        del output
                        error.update(reference=reference_name, relative_l2_limit=tolerance,
                                     passed=error["relative_l2"] <= tolerance)
                        if not error["passed"]:
                            raise AssertionError(f"{scope}/{backend}/{length}: output mismatch {error}")
                        record[scope][backend]["correctness"] = error
                    del reference
                result["lengths"].append(record)
                del query, key, value
    finally:
        del benchmark_model
    return result


if __name__ == "__main__" and sys.argv[1:] == ["--counters"]:
    _counter_child()
