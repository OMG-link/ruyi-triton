import pytest
import torch
import triton
import triton.language as tl


@triton.jit
def _shuffle_1d_kernel(x_ptr, y_ptr, out_ptr, N1: tl.constexpr, N2: tl.constexpr,
                       M: tl.constexpr, MASK: tl.constexpr):
    a = tl.load(x_ptr + tl.arange(0, N1))
    b = tl.load(y_ptr + tl.arange(0, N2))
    tl.store(out_ptr + tl.arange(0, M), tl.dev.shuffle(a, b, MASK))


@triton.jit
def _shuffle_2d_kernel(x_ptr, y_ptr, out_ptr, R1: tl.constexpr, R2: tl.constexpr,
                       C: tl.constexpr, M: tl.constexpr, MASK: tl.constexpr):
    cols = tl.arange(0, C)[None, :]
    a = tl.load(x_ptr + tl.arange(0, R1)[:, None] * C + cols)
    b = tl.load(y_ptr + tl.arange(0, R2)[:, None] * C + cols)
    tl.store(out_ptr + tl.arange(0, M)[:, None] * C + cols, tl.dev.shuffle(a, b, MASK))


@triton.jit
def _slide_kernel(x_ptr, out_ptr, N: tl.constexpr, STEPS: tl.constexpr, SLIDE: tl.constexpr):
    # A register slid down by one element per step, shifting in zeros: step s
    # stores what was element s.
    v = tl.load(x_ptr + tl.arange(0, N))
    zero = tl.zeros((N, ), v.dtype)
    for s in tl.static_range(STEPS):
        tl.store(out_ptr + s * N + tl.arange(0, N), v)
        v = tl.dev.shuffle(v, zero, SLIDE)


def _reference(a, b, mask):
    cat = torch.cat([a, b])
    return torch.stack([cat[i] for i in mask])


@pytest.mark.parametrize("n1, n2, mask", [
    (8, 4, (11, 0, 3, 8, 7, 2, 9, 1)),
    (64, 64, tuple(range(1, 65))),
    (4096, 4096, tuple(range(1, 4097))),  # too large for registers
])
def test_shuffle_1d(device, n1, n2, mask):
    a = torch.randn(n1, device=device)
    b = torch.randn(n2, device=device)
    out = torch.empty(len(mask), device=device)
    _shuffle_1d_kernel[(1, )](a, b, out, N1=n1, N2=n2, M=len(mask), MASK=mask)
    torch.testing.assert_close(out, _reference(a, b, mask), rtol=0, atol=0)


def test_shuffle_rows(device):
    mask = (5, 0, 4, 3, 1, 2, 4, 0)
    a = torch.randn(4, 8, device=device)
    b = torch.randn(2, 8, device=device)
    out = torch.empty(len(mask), 8, device=device)
    _shuffle_2d_kernel[(1, )](a, b, out, R1=4, R2=2, C=8, M=len(mask), MASK=mask)
    torch.testing.assert_close(out, _reference(a, b, mask), rtol=0, atol=0)


def test_shuffle_poison(device):
    # Only the defined elements are checked.
    a = torch.randn(8, device=device)
    out = torch.empty(8, device=device)
    _shuffle_1d_kernel[(1, )](a, a, out, N1=8, N2=8, M=8, MASK=(1, 2, 3, 4, 5, 6, 7, -1))
    torch.testing.assert_close(out[:7], a[1:], rtol=0, atol=0)


def test_slide_fp16(device):
    n, steps = 64, 8
    x = (torch.arange(n, device=device) + 1).half()
    out = torch.empty(steps, n, device=device, dtype=torch.float16)
    _slide_kernel[(1, )](x, out, N=n, STEPS=steps, SLIDE=tuple(range(1, n + 1)))
    ref = torch.stack([torch.cat([x[s:], torch.zeros(s, dtype=x.dtype)]) for s in range(steps)])
    torch.testing.assert_close(out, ref, rtol=0, atol=0)


@pytest.mark.parametrize("mask, match", [
    ((0, 9), r"out of range \[-1, 8\)"),
    ((), "non-empty list"),
])
def test_shuffle_invalid_mask(device, mask, match):
    a = torch.randn(4, device=device)
    out = torch.empty(max(len(mask), 1), device=device)
    with pytest.raises(Exception, match=match):
        _shuffle_1d_kernel[(1, )](a, a, out, N1=4, N2=4, M=max(len(mask), 1), MASK=mask)
