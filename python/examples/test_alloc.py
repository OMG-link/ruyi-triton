import os
import subprocess
import sys

import pytest
import torch

import triton
import triton.language as tl


@triton.jit
def _stage_chunks_kernel(x_ptr, out_ptr, NCHUNK: tl.constexpr, BLOCK: tl.constexpr):
    # Each program copies its NCHUNK x BLOCK row into scratch chunk by chunk,
    # then reads the chunks back in reverse order.
    pid = tl.program_id(0)
    buf = tl.alloc(NCHUNK * BLOCK, tl.float32)
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
    acc = tl.alloc((M, N), tl.float32)
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
    buf = tl.alloc(4 * BLOCK, tl.int32, alignment=128)
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
    buf = tl.alloc(BLOCK, tl.float32, placement=PLACEMENT)
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


def test_alloc_placement_spm(device):
    # On a board with a scratchpad the round trip works; on any other machine
    # the kernel compiles with a warning and fails at launch.
    from triton.backends.triton_shared import spm

    if isinstance(spm.runtime_driver(), spm.NoSpmDriver):
        with pytest.warns(UserWarning, match="no SPM driver"):
            with pytest.raises(spm.SpmOutOfMemoryError, match="no SPM driver"):
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
# Scratchpad drivers. These run only on a board with an SPM driver.
# --------------------------------------------------------------------------

def _spm_driver():
    from triton.backends.triton_shared import spm

    driver = spm.runtime_driver()
    if isinstance(driver, spm.NoSpmDriver):
        pytest.skip("no SPM driver on this board")
    return spm, driver


@triton.jit
def _spm_big_kernel(out_ptr, N: tl.constexpr):
    buf = tl.alloc(N, tl.int8, placement="spm")
    offs = tl.arange(0, 64)
    tl.store(buf + offs, offs.to(tl.int8))
    tl.store(out_ptr + offs, tl.load(buf + offs))


def test_spm_capacity(device):
    spm, driver = _spm_driver()
    out = torch.empty(64, dtype=torch.int8)
    with pytest.raises(spm.SpmOutOfMemoryError, match="can use at most"):
        _spm_big_kernel[(1,)](out, N=driver.budget_per_program + 4096)


def test_spm_full_budget(device):
    # The whole budget of a program instance, two K3 blocks.
    _, driver = _spm_driver()
    out = torch.empty(64, dtype=torch.int8)
    _spm_big_kernel[(1,)](out, N=driver.budget_per_program)
    torch.testing.assert_close(out, torch.arange(64, dtype=torch.int8))


def test_spm_restores_affinity(device):
    _spm_driver()
    before = os.sched_getaffinity(0)
    _launch_spm(device)
    assert os.sched_getaffinity(0) == before


_HOLD_K1 = """
import mmap, os, sys, time
fd = os.open("/dev/tcm", os.O_RDWR | os.O_SYNC)
m = mmap.mmap(fd, 512 << 10)
print("held", flush=True)
time.sleep(30)
"""

_HOLD_K3 = """
import ctypes, os, sys, time
cpu = int(sys.argv[1])
os.sched_setaffinity(0, {cpu})
lib = ctypes.CDLL("libspine_tcm.so")
lib.spine_tcm_runtime_mem_try_wait.restype = ctypes.c_void_p
block = 2 * ((cpu - 8) // 2)
assert lib.spine_tcm_runtime_mem_try_wait(block, 0)
print("held", flush=True)
time.sleep(30)
lib.spine_tcm_runtime_mem_release(block)
"""


def test_spm_occupied(device):
    # Another live process holds this core's scratchpad: the launch fails and
    # says so, without taking the scratchpad away from it.
    spm, driver = _spm_driver()
    cpu = sorted(os.sched_getaffinity(0))[0]
    if isinstance(driver, spm.SpacemitK3TcmDriver):
        script, args, expected = _HOLD_K3, [str(cpu)], "held by live threads"
    else:
        script, args, expected = _HOLD_K1, [], "mapped by another process"
    holder = subprocess.Popen([sys.executable, "-c", script, *args], stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "held"
        os.sched_setaffinity(0, {cpu})
        with pytest.raises(spm.SpmOutOfMemoryError, match=expected):
            _launch_spm(device)
    finally:
        holder.kill()
        holder.wait()
        if isinstance(driver, spm.SpacemitK3TcmDriver):
            spm.release_stale_k3()


def test_spm_k1_x60_core(device):
    # An X60 core has no TCM; the launch is refused instead of crashing.
    spm, driver = _spm_driver()
    if not isinstance(driver, spm.SpacemitK1TcmDriver):
        pytest.skip("K1 only")
    before = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {4})
    try:
        with pytest.raises(spm.SpmOutOfMemoryError, match="CPU 4 has no scratchpad"):
            _launch_spm(device)
    finally:
        os.sched_setaffinity(0, before)
