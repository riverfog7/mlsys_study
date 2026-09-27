"""File-backed Triton kernel; imported after the child process configures caches."""
import triton
import triton.language as tl


@triton.jit
def add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    value = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    tl.store(y_ptr + offsets, value + 1.0, mask=mask)
