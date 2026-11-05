import subprocess

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


def test_alloc_placement_spm_unimplemented(device):
    x = torch.randn(1, 64, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    # triton-shared-opt reports "placement 'spm' is not implemented yet".
    with pytest.raises(subprocess.CalledProcessError):
        _alloc_placement_kernel[(1,)](x, out, BLOCK=64, PLACEMENT="spm")


def test_alloc_placement_invalid(device):
    x = torch.randn(1, 64, device=device, dtype=torch.float32)
    out = torch.empty_like(x)
    with pytest.raises(Exception, match="placement must be 'global', 'local' or 'spm'"):
        _alloc_placement_kernel[(1,)](x, out, BLOCK=64, PLACEMENT="heap")
