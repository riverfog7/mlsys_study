"""Inspect actual file-backed CPython bytecode; model a tiny arithmetic stack."""
from study02.runtime import experiment_main


def run(args, out):
    import dis
    import inspect
    import io
    import contextlib
    import torch
    from study02.examples import stack_example, tiny

    instructions = [dict(offset=i.offset, opname=i.opname, argrepr=i.argrepr,
                         line=i.starts_line) for i in dis.get_instructions(stack_example)]
    stack, local, steps = [], {"x": 3}, []
    for instruction in dis.get_instructions(stack_example):
        op = instruction.opname
        if op == "LOAD_FAST":
            stack.append(local[instruction.argval])
        elif op == "LOAD_CONST":
            stack.append(instruction.argval)
        elif op == "STORE_FAST":
            local[instruction.argval] = stack.pop()
        elif op in {"BINARY_ADD", "BINARY_MULTIPLY", "BINARY_OP"}:
            right, left = stack.pop(), stack.pop()
            symbol = instruction.argrepr if op == "BINARY_OP" else ("+" if op == "BINARY_ADD" else "*")
            if symbol not in {"+", "*"}:
                raise ValueError(f"Stack illustration only supports + and *: {symbol}")
            stack.append(left + right if symbol == "+" else left * right)
        elif op == "RETURN_VALUE":
            returned = stack.pop()
        elif op not in {"RESUME", "CACHE", "NOP"}:
            raise ValueError(f"Unsupported teaching opcode in this Python: {op}")
        steps.append({"instruction": f"{instruction.offset}: {op} {instruction.argrepr}",
                      "stack": list(stack), "locals": dict(local)})
    assert returned == stack_example(3)
    frame = inspect.currentframe()
    frame_summary = {"function": frame.f_code.co_name, "filename": frame.f_code.co_filename,
                     "local_names": sorted(frame.f_locals), "caller": frame.f_back.f_code.co_name}
    del frame
    stream = io.StringIO()
    with contextlib.redirect_stdout(stream):
        dis.dis(tiny)
    (out / "raw" / "tiny_bytecode.txt").write_text(stream.getvalue())
    x = torch.arange(8., device="cuda").reshape(2, 4)
    weight = torch.arange(12., device="cuda").reshape(4, 3) / 10
    return {"instructions": instructions, "steps": steps, "returned": returned,
            "stack_note": "실제 dis 명령을 해석한 교육용 모델; CPython value stack 실측 아님",
            "code_object": {"varnames": list(tiny.__code__.co_varnames),
                            "constants": [repr(x) for x in tiny.__code__.co_consts],
                            "bytecode_hex": tiny.__code__.co_code.hex()},
            "frame": frame_summary, "tiny_output": tiny(x, weight).tolist()}


if __name__ == "__main__":
    experiment_main("python_runtime", run)
