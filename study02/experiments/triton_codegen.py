"""Compile and execute Triton, then save this compilation's actual IR and binary."""
from study02.runtime import experiment_main


def configure(parser):
    parser.add_argument("--elements", type=int, default=529)
    parser.add_argument("--block", type=int, default=256)
    parser.add_argument("--nsight", action="store_true", help="Capture one kernel's hardware counters using ncu")


def run(args, out):
    import csv
    import io
    import shutil
    import subprocess
    import sys
    import torch
    import triton
    from study02.kernels import add_one_kernel

    if args.elements < 1 or args.block < 1 or args.block & (args.block - 1):
        raise ValueError("elements must be positive and block must be a positive power of two")
    x = torch.randn(args.elements, device="cuda", dtype=getattr(torch, args.dtype))
    y = torch.empty_like(x)
    compiled = add_one_kernel[(triton.cdiv(args.elements, args.block),)](x, y, args.elements, BLOCK=args.block)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, x + 1)
    stages = {}
    for stage in ("ttir", "ttgir", "llir", "ptx", "cubin"):
        content = compiled.asm.get(stage)
        if content is None:
            stages[stage] = {"status": "unavailable", "reason": "not provided by this Triton compiler"}
            continue
        path = out / "raw" / f"add_one.{stage}"
        if isinstance(content, bytes):
            path.write_bytes(content)
        else:
            path.write_text(content, encoding="utf-8")
        stages[stage] = {"status": "generated", "path": str(path.relative_to(out)), "bytes": path.stat().st_size}
    disassembler = shutil.which("nvdisasm") or shutil.which("cuobjdump")
    if disassembler and stages["cubin"]["status"] == "generated":
        command = [disassembler]
        if "cuobjdump" in disassembler:
            command.append("--dump-sass")
        command.append(str(out / stages["cubin"]["path"]))
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
        (out / "raw" / "disassembler.stderr.txt").write_text(result.stderr)
        if result.returncode == 0:
            (out / "raw" / "add_one.sass").write_text(result.stdout)
            stages["sass"] = {"status": "generated", "path": "raw/add_one.sass"}
        else:
            stages["sass"] = {"status": "unavailable", "reason": result.stderr[-1500:]}
    else:
        stages["sass"] = {"status": "unavailable", "reason": "nvdisasm/cuobjdump or cubin is unavailable"}
    counters = {"status": "not_measured", "tool": shutil.which("ncu"),
                "reason": "Set nsight=True to run Nsight Compute with hardware-counter permissions."}
    if args.nsight:
        if not counters["tool"]:
            counters.update(status="unavailable", reason="ncu is not installed")
        else:
            command = [counters["tool"], "--target-processes", "all", "--launch-count", "1",
                       "--kernel-name", "regex:add_one_kernel", "--csv", "--metrics",
                       "sm__warps_active.avg.pct_of_peak_sustained_active,dram__bytes_read.sum,dram__bytes_write.sum",
                       sys.executable, "-m", "study02.experiments.triton_codegen",
                       "--output", str(out / "raw" / "nsight-child"), "--elements", str(args.elements),
                       "--block", str(args.block), "--dtype", args.dtype, "--seed", str(args.seed)]
            try:
                measured = subprocess.run(command, capture_output=True, text=True, timeout=120)
                (out / "raw" / "nsight.stdout.txt").write_text(measured.stdout)
                (out / "raw" / "nsight.stderr.txt").write_text(measured.stderr)
                lines = measured.stdout.splitlines()
                header = next((i for i, line in enumerate(lines) if '"Metric Name"' in line), None)
                if measured.returncode == 0 and header is not None:
                    rows = list(csv.DictReader(io.StringIO("\n".join(lines[header:]))))
                    metrics = [{"name": row["Metric Name"], "unit": row.get("Metric Unit", ""),
                                "value": float(row["Metric Value"].replace(",", ""))}
                               for row in rows if row.get("Metric Value") and row.get("Metric Name")]
                    counters.update(status="measured", metrics=metrics, reason="", command=command)
                else:
                    counters.update(status="unavailable", command=command,
                                    reason=(measured.stdout + measured.stderr)[-3000:])
            except subprocess.TimeoutExpired:
                counters.update(status="unavailable", reason="Nsight exceeded the 120-second limit", command=command)
    return {"elements": args.elements, "block": args.block,
            "programs": triton.cdiv(args.elements, args.block), "stages": stages,
            "resources": {"registers_per_thread": compiled.n_regs,
                          "shared_memory_bytes": compiled.metadata.shared,
                          "num_warps": compiled.metadata.num_warps},
            "sample_input": x[:12].tolist(), "sample_output": y[:12].tolist(),
            "correctness": {"passed": True},
            "hardware_counters": counters}


if __name__ == "__main__":
    experiment_main("triton_codegen", run, configure)
