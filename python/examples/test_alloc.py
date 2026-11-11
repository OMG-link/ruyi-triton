import os
import subprocess
import sys
import warnings

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _stage_chunks_kernel(x_ptr, out_ptr, NCHUNK: tl.constexpr, BLOCK: tl.constexpr):
    # Each program copies its NCHUNK x BLOCK row into scratch chunk by chunk,
    # then reads the chunks back in reverse order.
    pid = tl.program_id(0)
    buf = tl.dev.alloc(NCHUNK * BLOCK, tl.float32)
    offs = tl.arange(0, BLOCK)
    row = pid * NCHUNK * BLOCK
    for i in range(NCHUNK):
        x = tl.load(x_ptr + row + i * BLOCK + offs)
        tl.store(buf + i * BLOCK + offs, x * 2.0)
    for i in range(NCHUNK):
        y = tl.load(buf + (NCHUNK - 1 - i) * BLOCK + offs)
        tl.store(out_ptr + row + i * BLOCK + offs, y)


def test_stage_chunks(device):
    nprog, nchunk, block = 5, 4, 32
    x = torch.randn(nprog, nchunk, block, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    _stage_chunks_kernel[(nprog,)](x, out, NCHUNK=nchunk, BLOCK=block)
    torch.testing.assert_close(out, (x * 2.0).flip(1))


@triton.jit
def _scratch_accumulator_kernel(x_ptr, out_ptr, K, M: tl.constexpr, N: tl.constexpr):
    # A 2-D accumulator lives in scratch across loop iterations: each
    # iteration loads it, adds one input tile and stores it back.
    pid = tl.program_id(0)
    acc = tl.dev.alloc((M, N), tl.float32)
    om = tl.arange(0, M)[:, None]
    on = tl.arange(0, N)[None, :]
    tl.store(acc + om * N + on, tl.zeros((M, N), tl.float32))
    base = x_ptr + pid * K * M * N
    for k in range(K):
        a = tl.load(acc + om * N + on)
        t = tl.load(base + k * M * N + om * N + on)
        tl.store(acc + om * N + on, a + t)
    # Read the accumulator back transposed through strided offsets.
    r = tl.load(acc + tl.arange(0, N)[:, None] + tl.arange(0, M)[None, :] * N)
    tl.store(out_ptr + pid * M * N + tl.arange(0, N)[:, None] * M + tl.arange(0, M)[None, :], r)


@pytest.mark.parametrize("dtype", [torch.float32])
def test_scratch_accumulator(device, dtype):
    nprog, k, m, n = 3, 6, 8, 16
    x = torch.randn(nprog, k, m, n, device=device, dtype=dtype)
    out = torch.empty(nprog, n, m, device=device, dtype=dtype)
    _scratch_accumulator_kernel[(nprog,)](x, out, k, M=m, N=n)
    torch.testing.assert_close(out, x.sum(1).transpose(1, 2), rtol=1e-5, atol=1e-5)


@triton.jit
def _scratch_int_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    # Integer scratch with an explicit alignment, written at a program-dependent
    # offset inside the buffer.
    pid = tl.program_id(0)
    buf = tl.dev.alloc(4 * BLOCK, tl.int32, alignment=128)
    offs = tl.arange(0, BLOCK)
    x = tl.load(x_ptr + pid * BLOCK + offs)
    slot = pid % 4
    tl.store(buf + slot * BLOCK + offs, x + pid)
    y = tl.load(buf + slot * BLOCK + offs)
    tl.store(out_ptr + pid * BLOCK + offs, y)


def test_scratch_int(device):
    nprog, block = 7, 64
    x = torch.randint(-1000, 1000, (nprog, block), device=device, dtype=torch.int32)
    out = torch.empty_like(x)
    _scratch_int_kernel[(nprog,)](x, out, BLOCK=block)
    ref = x + torch.arange(nprog, dtype=torch.int32)[:, None]
    torch.testing.assert_close(out, ref)


@triton.jit
def _alloc_placement_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr, PLACEMENT: tl.constexpr):
    # A round trip through scratch at the requested placement.
    pid = tl.program_id(0)
    buf = tl.dev.alloc(BLOCK, tl.float32, placement=PLACEMENT)
    offs = tl.arange(0, BLOCK)
    tl.store(buf + offs, tl.load(x_ptr + pid * BLOCK + offs) + 1.0)
    tl.store(out_ptr + pid * BLOCK + offs, tl.load(buf + offs))


@pytest.mark.parametrize("placement", ["global", "local"])
def test_alloc_placement(device, placement):
    nprog, block = 3, 64
    x = torch.randn(nprog, block, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    _alloc_placement_kernel[(nprog,)](x, out, BLOCK=block, PLACEMENT=placement)
    torch.testing.assert_close(out, x + 1.0)


def _launch_spm(device):
    x = torch.randn(3, 64, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    _alloc_placement_kernel[(3,)](x, out, BLOCK=64, PLACEMENT="spm")
    return x, out


def _board_config():
    from triton.backends.triton_shared import board, compiler

    backend = compiler.CPUBackend(triton.runtime.driver.active.get_current_target())
    return board.CONFIGS[backend.parse_options({}).board_config]


def test_alloc_placement_spm(device):
    # On a board configuration with a scratchpad the round trip works; on any
    # other one the kernel compiles with a warning and fails at launch.
    from triton.backends.triton_shared import spm

    if not _board_config().spm_bytes_per_cpu:
        # The compile-time warning only appears when the kernel is compiled,
        # not when it comes from the cache.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(spm.SpmOutOfMemoryError, match="has no scratchpad"):
                _launch_spm(device)
    else:
        x, out = _launch_spm(device)
        torch.testing.assert_close(out, x + 1.0)


def test_alloc_placement_invalid(device):
    x = torch.randn(1, 64, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    with pytest.raises(Exception, match="placement must be 'global', 'local' or 'spm'"):
        _alloc_placement_kernel[(1,)](x, out, BLOCK=64, PLACEMENT="heap")


# --------------------------------------------------------------------------
# Scratchpad. These run only with a board configuration that has one. The
# runtime holds the whole scratchpad from the first launch of the process on;
# a program instance of a kernel without parallel region runs on one CPU and
# gets that CPU's share.
# --------------------------------------------------------------------------


def _spm_config():
    from triton.backends.triton_shared import spm

    config = _board_config()
    if not config.spm_bytes_per_cpu:
        pytest.skip("no scratchpad in this board configuration")
    return spm, config


@triton.jit
def _spm_big_kernel(out_ptr, N: tl.constexpr):
    buf = tl.dev.alloc(N, tl.int8, placement="spm")
    offs = tl.arange(0, 64)
    tl.store(buf + offs, offs.to(tl.int8))
    tl.store(out_ptr + offs, tl.load(buf + offs))


def test_spm_capacity(device):
    spm, config = _spm_config()
    out = torch.empty(64, dtype=torch.int8)
    with pytest.raises(spm.SpmOutOfMemoryError, match="gets only"):
        _spm_big_kernel[(1,)](out, N=config.spm_bytes_per_cpu + 4096)


def test_spm_full_share(device):
    # The whole share of the CPU running the program instance.
    _, config = _spm_config()
    out = torch.empty(64, dtype=torch.int8)
    _spm_big_kernel[(1,)](out, N=config.spm_bytes_per_cpu)
    torch.testing.assert_close(out, torch.arange(64, dtype=torch.int8))


def test_spm_keeps_caller_affinity(device):
    _spm_config()
    before = os.sched_getaffinity(0)
    _launch_spm(device)
    assert os.sched_getaffinity(0) == before


_CHILD = """
import torch, triton, triton.language as tl
from triton.backends.triton_shared import spm
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def kernel(x_ptr, out_ptr):
    buf = tl.dev.alloc(64, tl.float32, placement="spm")
    offs = tl.arange(0, 64)
    tl.store(buf + offs, tl.load(x_ptr + offs))
    tl.store(out_ptr + offs, tl.load(buf + offs))

x = torch.randn(64)
out = torch.empty_like(x)
try:
    kernel[(1,)](x, out)
    print("launched")
except spm.SpmOutOfMemoryError as e:
    print("refused:", e)
"""


def _run_child(tmp_path, env=None):
    script = tmp_path / "child.py"
    script.write_text(_CHILD)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            env={**os.environ, **(env or {})}, timeout=600)
    assert result.returncode == 0, result.stderr
    return result.stdout


def test_spm_occupied(device, tmp_path):
    # This process holds the scratchpad (from its first launch on); another
    # process is refused and told who holds it, without taking it away.
    spm, config = _spm_config()
    _launch_spm(device)
    expected = ("held by another process"
                if config.spm_driver is spm.SpacemitK3TcmDriver
                else "mapped by another process")
    out = _run_child(tmp_path)
    assert "refused:" in out and expected in out, out
    x, out_tensor = _launch_spm(device)
    torch.testing.assert_close(out_tensor, x + 1.0)


def test_spm_cpu_without_share(device, tmp_path):
    # On the K1, the X60 core 4 has no TCM: the launch is refused instead of
    # crashing.
    spm, config = _spm_config()
    if config.spm_driver is not spm.SpacemitK1TcmDriver:
        pytest.skip("K1 only")
    out = _run_child(tmp_path, {"TRITON_CPAR_CPUS": "4"})
    assert "without a scratchpad share" in out, out
