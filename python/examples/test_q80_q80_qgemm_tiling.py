"""Integration test for Q8_0 × Q8_0 QGEMM with qgemm_opt_enable=1.

Verifies that the QGEMM multi-level tiling optimization produces numerically
correct results by comparing Triton kernel output against a PyTorch reference
implementation. The test exercises the full compilation pipeline including:
  - decompose-high-level-ops (TTIR → Linalg)
  - qgemm-tiling (two levels: tile-m=32 tile-n=32 then tile-m=4 tile-n=4)
  - lower-linalg-to-vir → lower-vir-to-vector → ... → native .o
"""

import os


# Force recompilation so the QGEMM tiling pass is exercised regardless of cache state.
os.environ["TRITON_ALWAYS_COMPILE"] = "1"

import pytest
import torch
import triton
import triton.language as tl


QK8_0 = 32
TL_QK8_0 = tl.constexpr(QK8_0)


# ---------------------------------------------------------------------------
# PyTorch reference implementation
# ---------------------------------------------------------------------------

def q80_q80_reference(a_q, a_d, w_q, w_d, M, N, K):
    """PyTorch reference for Q8_0 × Q8_0 quantized matmul."""
    QK = QK8_0

    a_d = a_d.unsqueeze(2).expand(M, K // QK, QK).reshape(M, K)
    w_d = w_d.unsqueeze(1).expand(K // QK, QK, N).reshape(K, N)

    a = a_q.to(torch.float32) * a_d.to(torch.float32)
    w = w_q.to(torch.float32) * w_d.to(torch.float32)

    c = a @ w
    return c


# ---------------------------------------------------------------------------
# Triton kernel: Q8_0 × Q8_0 canonicalized form
# ---------------------------------------------------------------------------

@triton.jit
def q80_q80_canonicalized(
    a_q_ptr,    # int8    (M, K)
    a_d_ptr,    # float16 (M, K // QK)
    w_q_ptr,    # int8    (K, N)
    w_d_ptr,    # float16 (K // QK, N)
    c_ptr,      # float32 (M, N)
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
):
    QK: tl.constexpr = TL_QK8_0
    KB: tl.constexpr = K // QK

    pid = tl.program_id(0)
    n_per_pid: tl.constexpr = N // T
    n_start = pid * n_per_pid

    tl.device_assert(tl.num_programs(0) == T, "grid.x must equal T")
    tl.device_assert(tl.num_programs(1) == 1, "grid.y must be 1")
    tl.device_assert(tl.num_programs(2) == 1, "grid.z must be 1")
    tl.device_assert(tl.num_programs(0) * n_per_pid == N, "N must be divisible by T")

    a_q_block_ptr = tl.make_block_ptr(
        base=a_q_ptr,
        shape=(M, KB, QK),
        strides=(K, QK, 1),
        offsets=(0, 0, 0),
        block_shape=(M, KB, QK),
        order=(2, 1, 0),
    )
    a_d_block_ptr = tl.make_block_ptr(
        base=a_d_ptr,
        shape=(M, KB),
        strides=(KB, 1),
        offsets=(0, 0),
        block_shape=(M, KB),
        order=(1, 0),
    )
    w_q_block_ptr = tl.make_block_ptr(
        base=w_q_ptr,
        shape=(KB, QK, N),
        strides=(QK * N, N, 1),
        offsets=(0, 0, n_start),
        block_shape=(KB, QK, n_per_pid),
        order=(2, 1, 0),
    )
    w_d_block_ptr = tl.make_block_ptr(
        base=w_d_ptr,
        shape=(KB, N),
        strides=(N, 1),
        offsets=(0, n_start),
        block_shape=(KB, n_per_pid),
        order=(1, 0),
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=(M, N),
        strides=(N, 1),
        offsets=(0, n_start),
        block_shape=(M, n_per_pid),
        order=(1, 0),
    )

    a_q = tl.load(a_q_block_ptr)                          # (M, KB, QK)
    a_q = tl.expand_dims(a_q, axis=1)                     # (M, 1, KB, QK)
    a_q = tl.broadcast_to(a_q, (M, n_per_pid, KB, QK))   # (M, n_per_pid, KB, QK)

    w_q = tl.load(w_q_block_ptr)                          # (KB, QK, n_per_pid)
    w_q = tl.expand_dims(w_q, axis=0)                     # (1, KB, QK, n_per_pid)
    w_q = tl.broadcast_to(w_q, (M, KB, QK, n_per_pid))   # (M, KB, QK, n_per_pid)
    w_q = tl.trans(w_q, 0, 3, 1, 2)                       # (M, n_per_pid, KB, QK)

    a_d = tl.load(a_d_block_ptr)                          # (M, KB)
    a_d = tl.expand_dims(a_d, axis=1)                     # (M, 1, KB)
    a_d = tl.broadcast_to(a_d, (M, n_per_pid, KB))       # (M, n_per_pid, KB)

    w_d = tl.load(w_d_block_ptr)                          # (KB, n_per_pid)
    w_d = tl.expand_dims(w_d, axis=0)                     # (1, KB, n_per_pid)
    w_d = tl.broadcast_to(w_d, (M, KB, n_per_pid))       # (M, KB, n_per_pid)
    w_d = tl.trans(w_d, 0, 2, 1)                          # (M, n_per_pid, KB)

    block_sum = (a_q.to(tl.int16) * w_q.to(tl.int16))    # (M, n_per_pid, KB, QK)
    block_sum = tl.sum(block_sum.to(tl.int32), axis=3)    # (M, n_per_pid, KB)
    d = a_d.to(tl.float32) * w_d.to(tl.float32)          # (M, n_per_pid, KB)
    c = tl.sum(block_sum.to(tl.float32) * d, axis=2)     # (M, n_per_pid)

    tl.store(c_block_ptr, c)


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

# Test parameters: (M, N, K, T)
# Shapes exercise multi-level tiling (tile-m=32 tile-n=32 → tile-m=4 tile-n=4)
_TEST_CASES = [
    (64, 32, 64, 1),    # Outer: M tiled (64→32), N not tiled (32==32)
                         # Inner: M tiled (32→4), N tiled (32→4)
    (64, 64, 64, 2),    # Outer: M tiled (64→32), N tiled (64→32)
                         # Inner: M tiled (32→4), N tiled (32→4)
    (128, 64, 64, 2),   # Larger: M tiled (128→32), N tiled (64→32)
                         # Inner: M tiled (32→4), N tiled (32→4)
    (32, 32, 32, 1),    # M <= tile-m, N <= tile-n → no tiling applied
    (64, 96, 64, 3),    # Larger N with multiple programs
]


@pytest.mark.parametrize("M,N,K,T", _TEST_CASES)
def test_q80_q80_qgemm_tiling(device, M, N, K, T):
    """Verify Q8_0 × Q8_0 QGEMM produces correct results with QGEMM tiling enabled.

    Compares the Triton kernel output (compiled with --qgemm-tiling=tile-m=32 tile-n=32)
    against a PyTorch reference implementation. The test verifies numerical equivalence
    using max-abs-diff tolerance.
    """
    QK = QK8_0
    assert K % QK == 0, "K must be divisible by QK"
    assert N % T == 0, "N must be divisible by T"

    torch.manual_seed(42)

    # Generate quantized input data
    a_q = torch.randint(-8, 8, (M, K), dtype=torch.int8)
    w_q = torch.randint(-8, 8, (K, N), dtype=torch.int8)
    a_d = torch.randn(M, K // QK, dtype=torch.float16)
    w_d = torch.randn(K // QK, N, dtype=torch.float16)

    # PyTorch reference
    ref = q80_q80_reference(a_q, a_d, w_q, w_d, M, N, K).to(torch.float32)

    # Triton kernel output
    out = torch.empty((M, N), dtype=torch.float32)
    grid = (T, 1, 1)
    q80_q80_canonicalized[grid](
        a_q, a_d, w_q, w_d, out,
        M=M, N=N, K=K, T=T,
        qgemm_opt_enable=1,
    )

    # Verify numerical correctness
    max_abs_diff = (out - ref).abs().max().item()
    assert max_abs_diff < 1e-2, (
        f"Numerical mismatch with qgemm_opt_enable=1: "
        f"max_abs_diff={max_abs_diff:.6g} (threshold=1e-2) "
        f"for M={M}, N={N}, K={K}, T={T}"
    )

    # Also check with torch.allclose for relative tolerance
    assert torch.allclose(out, ref, atol=1e-2, rtol=1e-2), (
        f"torch.allclose failed with qgemm_opt_enable=1 "
        f"for M={M}, N={N}, K={K}, T={T}"
    )
