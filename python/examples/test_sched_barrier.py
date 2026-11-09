import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch

import triton
import triton.language as tl
from triton.compiler.errors import CompilationError
from test_utils import get_llvm_bin_path


def _tile():
    """Elements of an f32 tile that fills the smallest register group.

    The smallest e32 group on RISC-V is LMUL = 1/2 (ELEN 64), VLEN / 64 f32
    elements; on other hosts four f32 fill one 128-bit register.
    """
    from triton._C.libtriton import triton_shared

    vlen = triton_shared.read_host_vlen()
    return max(4, vlen // 64)


N = _tile()


@triton.jit
def _single_barrier_kernel(x_ptr, out_ptr, MEMORY: tl.constexpr, N: tl.constexpr):
    offs = tl.arange(0, N)
    x = tl.load(x_ptr + offs)
    x = tl.dev.sched_barrier(x, memory=MEMORY)
    tl.store(out_ptr + offs, x * 2.0)


@triton.jit
def _multiple_barrier_kernel(a_ptr, b_ptr, out_ptr, N: tl.constexpr):
    offs = tl.arange(0, N)
    a = tl.load(a_ptr + offs)
    b = tl.load(b_ptr + offs)
    a, b = tl.dev.sched_barrier(a, b)
    tl.store(out_ptr + offs, a + b)


@triton.jit
def _zero_argument_barrier_kernel(out_ptr):
    tl.dev.sched_barrier()
    tl.store(out_ptr, 1.0)


@triton.jit
def _non_tensor_barrier_kernel(x_ptr, out_ptr):
    offs = tl.arange(0, 4)
    x = tl.load(x_ptr + offs)
    x, _ = tl.dev.sched_barrier(x, 1)
    tl.store(out_ptr + offs, x)


@triton.jit
def _ragged_barrier_kernel(x_ptr, out_ptr):
    offs = tl.arange(0, 3)
    x = tl.load(x_ptr + offs)
    x = tl.dev.sched_barrier(x)
    tl.store(out_ptr + offs, x)


def test_sched_barrier_single_tensor(device):
    x = torch.randn(N, device=device)
    out = torch.empty_like(x)
    _single_barrier_kernel[(1,)](x, out, MEMORY=False, N=N)
    torch.testing.assert_close(out, x * 2.0)


def test_sched_barrier_multiple_tensors(device):
    a = torch.randn(N, device=device)
    b = torch.randn(N, device=device)
    out = torch.empty_like(a)
    _multiple_barrier_kernel[(1,)](a, b, out, N=N)
    torch.testing.assert_close(out, a + b)


@pytest.mark.parametrize("memory", [False, True])
def test_sched_barrier_emits_sideeffect_asm(device, memory):
    x = torch.randn(N, device=device)
    out = torch.empty_like(x)
    compiled = _single_barrier_kernel.warmup(x, out, MEMORY=memory, N=N, grid=(1,))
    llir = compiled.asm["llir"]
    assert "asm sideeffect" in llir
    assert ("~{memory}" in llir) is memory


@pytest.mark.parametrize(
    "kernel, args",
    [
        (_zero_argument_barrier_kernel, (torch.empty(1),)),
        (_non_tensor_barrier_kernel, (torch.empty(4), torch.empty(4))),
    ],
)
def test_sched_barrier_rejects_invalid_arguments(device, kernel, args):
    tensors = tuple(t.to(device) for t in args)
    expected = (
        "sched_barrier requires at least one tensor"
        if kernel is _zero_argument_barrier_kernel
        else "sched_barrier arguments must be tensors"
    )
    with pytest.raises(CompilationError, match=expected):
        kernel[(1,)](*tensors)


def test_sched_barrier_rejects_ragged_register_shape(device, capfd):
    x = torch.randn(3, device=device)
    out = torch.empty_like(x)
    with pytest.raises(subprocess.CalledProcessError):
        _ragged_barrier_kernel[(1,)](x, out)
    assert "cannot hold operand in registers" in capfd.readouterr().err


RISCV_USER_KERNEL = r"""
from pathlib import Path
import sys

import torch
import triton
import triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def barrier_kernel(x, y):
    offsets = tl.arange(0, 4)
    values = tl.load(x + offsets)
    values = tl.dev.sched_barrier(values)
    tl.store(y + offsets, values + 1.0)

x = torch.zeros((4,), dtype=torch.float32)
y = torch.empty_like(x)
compiled = barrier_kernel.warmup(x, y, grid=(1,), riscv_vlen=256)
Path(sys.argv[1]).write_bytes(compiled.asm["obj"])
Path(sys.argv[2]).write_text(compiled.asm["llir"])
"""


def test_sched_barrier_riscv_cross_compile_warmup(tmp_path):
    root = Path(__file__).resolve().parents[3]
    kernel_path = tmp_path / "sched_barrier_user_kernel.py"
    obj_path = tmp_path / "sched_barrier_user_kernel.o"
    llir_path = tmp_path / "sched_barrier_user_kernel.ll"
    kernel_path.write_text(RISCV_USER_KERNEL)

    env = os.environ.copy()
    env.update(
        {
            "TRITON_RISCV_TARGET_TRIPLE": "riscv64-unknown-linux-gnu",
            "TRITON_RISCV_MCPU": "spacemit-x60",
        }
    )
    subprocess.check_call(
        [sys.executable, str(kernel_path), str(obj_path), str(llir_path)],
        cwd=root,
        env=env,
    )
    assert obj_path.stat().st_size > 0
    header = subprocess.check_output(
        [get_llvm_bin_path("llvm-readelf"), "-h", str(obj_path)], text=True
    )
    assert "Machine:                           RISC-V" in header
    assert "asm sideeffect" in llir_path.read_text()
