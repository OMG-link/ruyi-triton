"""Property-based test for the decompose-high-level-ops pass.

This test verifies that the decompose pass (tt.dot → trans/expand_dims/broadcast/mulf/reduce)
produces numerically equivalent results to PyTorch's torch.matmul reference implementation.

The decompose pass is enabled by default via TRITON_ENABLE_QGEMM_OPT=1.
We run a Triton kernel with tl.dot through the full compilation pipeline
(which includes --decompose-high-level-ops) and compare against PyTorch.
"""

import os
import pytest
import torch
import triton
import triton.language as tl
from hypothesis import given, settings, assume, HealthCheck
from hypothesis import strategies as st


# Ensure the decompose pass is enabled
os.environ.setdefault("TRITON_ENABLE_QGEMM_OPT", "1")


# ---------------------------------------------------------------------------
# Triton kernels for single-block matmul (exercises tl.dot directly)
# ---------------------------------------------------------------------------

@triton.jit
def dot_kernel_f32(
    A, B, C_acc, Out,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Single-block matmul kernel: Out = dot(A, B) + C_acc (f32 inputs)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load A block [BLOCK_M, BLOCK_K]
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

    # Load B block [BLOCK_K, BLOCK_N]
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

    # Load accumulator [BLOCK_M, BLOCK_N]
    c_ptrs = C_acc + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

    # Compute dot product: result = dot(a, b) + c
    result = tl.dot(a, b, acc=c, out_dtype=tl.float32, allow_tf32=False)

    # Store result
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@triton.jit
def dot_kernel_f16_to_f32(
    A, B, C_acc, Out,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_om, stride_on,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    """Single-block matmul kernel: Out = dot(A_f16, B_f16) + C_f32 (f16 inputs, f32 output)."""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # Load A block [BLOCK_M, BLOCK_K] as f16
    a_ptrs = A + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K), other=0.0)

    # Load B block [BLOCK_K, BLOCK_N] as f16
    b_ptrs = B + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
    b = tl.load(b_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

    # Load accumulator [BLOCK_M, BLOCK_N] as f32
    c_ptrs = C_acc + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c = tl.load(c_ptrs, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N), other=0.0)

    # Compute dot product: result = dot(a_f16, b_f16) + c_f32 → f32 output
    result = tl.dot(a, b, acc=c, out_dtype=tl.float32, allow_tf32=False)

    # Store result
    out_ptrs = Out + offs_m[:, None] * stride_om + offs_n[None, :] * stride_on
    tl.store(out_ptrs, result, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


# ---------------------------------------------------------------------------
# Helper: run a single-block dot kernel and return the result
# ---------------------------------------------------------------------------

def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    p = 1
    while p < n:
        p *= 2
    return p


def run_dot_f32(M, N, K, a, b, c_acc):
    """Run the f32 dot kernel with the decompose pass enabled."""
    BLOCK_M = _next_power_of_2(M)
    BLOCK_N = _next_power_of_2(N)
    BLOCK_K = _next_power_of_2(K)

    out = torch.empty((M, N), device="cpu", dtype=torch.float32)

    grid = (1, 1)
    dot_kernel_f32[grid](
        a, b, c_acc, out,
        M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c_acc.stride(0), c_acc.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


def run_dot_f16_to_f32(M, N, K, a_f16, b_f16, c_acc_f32):
    """Run the f16→f32 dot kernel with the decompose pass enabled."""
    BLOCK_M = _next_power_of_2(M)
    BLOCK_N = _next_power_of_2(N)
    BLOCK_K = _next_power_of_2(K)

    out = torch.empty((M, N), device="cpu", dtype=torch.float32)

    grid = (1, 1)
    dot_kernel_f16_to_f32[grid](
        a_f16, b_f16, c_acc_f32, out,
        M, N, K,
        a_f16.stride(0), a_f16.stride(1),
        b_f16.stride(0), b_f16.stride(1),
        c_acc_f32.stride(0), c_acc_f32.stride(1),
        out.stride(0), out.stride(1),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )
    return out


# ---------------------------------------------------------------------------
# Hypothesis strategies
# ---------------------------------------------------------------------------

# Shape strategy: M, N, K ∈ [1, 128]
# We use powers of 2 for block sizes, but the actual matrix dimensions can be
# any value in [1, 128] since we use masking in the kernel.
shape_strategy = st.integers(min_value=1, max_value=128)


# ---------------------------------------------------------------------------
# Property Test: Numerical Equivalence (f32 → f32)
# ---------------------------------------------------------------------------

@given(
    M=shape_strategy,
    N=shape_strategy,
    K=shape_strategy,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_numerical_equivalence_f32(M, N, K, seed):
    """Property 1: Numerical Equivalence (f32 → f32).

    Feature: decompose-high-level-ops, Property 1: Numerical Equivalence
    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 4.1, 5.1, 7.4

    For any valid shapes M, N, K and random f32 tensor data,
    the decomposed dot operation shall produce results numerically
    equivalent to torch.matmul(a, b) + c.
    """
    torch.manual_seed(seed)

    a = torch.randn((M, K), device="cpu", dtype=torch.float32)
    b = torch.randn((K, N), device="cpu", dtype=torch.float32)
    c_acc = torch.randn((M, N), device="cpu", dtype=torch.float32)

    # Run through Triton with decompose pass
    triton_result = run_dot_f32(M, N, K, a, b, c_acc)

    # PyTorch reference: matmul(a, b) + c
    ref_result = torch.matmul(a, b) + c_acc

    # Compare with tolerance (f32 accumulation may have small rounding differences)
    torch.testing.assert_close(
        triton_result, ref_result,
        atol=1e-2, rtol=1e-3,
        msg=f"Numerical mismatch for f32 dot with M={M}, N={N}, K={K}",
    )


# ---------------------------------------------------------------------------
# Property Test: Numerical Equivalence (f16 → f32)
# ---------------------------------------------------------------------------

@given(
    M=shape_strategy,
    N=shape_strategy,
    K=shape_strategy,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_numerical_equivalence_f16_to_f32(M, N, K, seed):
    """Property 1: Numerical Equivalence (f16 inputs → f32 output).

    Feature: decompose-high-level-ops, Property 1: Numerical Equivalence
    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 4.1, 4.2, 4.3, 5.1, 7.4

    For any valid shapes M, N, K and random f16 tensor data,
    the decomposed dot operation (f16 inputs, f32 accumulation)
    shall produce results numerically equivalent to
    torch.matmul(a.float(), b.float()) + c.
    """
    torch.manual_seed(seed)

    # Use smaller values to avoid f16 overflow
    a_f16 = (torch.randn((M, K), device="cpu", dtype=torch.float32) * 0.5).to(torch.float16)
    b_f16 = (torch.randn((K, N), device="cpu", dtype=torch.float32) * 0.5).to(torch.float16)
    c_acc = torch.randn((M, N), device="cpu", dtype=torch.float32) * 0.5

    # Run through Triton with decompose pass
    triton_result = run_dot_f16_to_f32(M, N, K, a_f16, b_f16, c_acc)

    # PyTorch reference: matmul in f32 precision (matching the decompose behavior)
    # The decompose pass converts f16 → f32 before multiplication
    ref_result = torch.matmul(a_f16.float(), b_f16.float()) + c_acc

    # Compare with tolerance appropriate for f16→f32 computation
    torch.testing.assert_close(
        triton_result, ref_result,
        atol=1e-2, rtol=1e-3,
        msg=f"Numerical mismatch for f16→f32 dot with M={M}, N={N}, K={K}",
    )


# ---------------------------------------------------------------------------
# Property Test: Numerical Equivalence with zero accumulator (f32)
# ---------------------------------------------------------------------------

@given(
    M=shape_strategy,
    N=shape_strategy,
    K=shape_strategy,
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(
    max_examples=5,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_numerical_equivalence_f32_zero_acc(M, N, K, seed):
    """Property 1: Numerical Equivalence with zero accumulator (f32).

    Feature: decompose-high-level-ops, Property 1: Numerical Equivalence
    Validates: Requirements 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 5.1, 7.4

    For any valid shapes M, N, K and random f32 tensor data with zero accumulator,
    the decomposed dot operation shall produce results numerically
    equivalent to torch.matmul(a, b).
    """
    torch.manual_seed(seed)

    a = torch.randn((M, K), device="cpu", dtype=torch.float32)
    b = torch.randn((K, N), device="cpu", dtype=torch.float32)
    c_acc = torch.zeros((M, N), device="cpu", dtype=torch.float32)

    # Run through Triton with decompose pass
    triton_result = run_dot_f32(M, N, K, a, b, c_acc)

    # PyTorch reference: matmul(a, b) + 0 = matmul(a, b)
    ref_result = torch.matmul(a, b)

    torch.testing.assert_close(
        triton_result, ref_result,
        atol=1e-2, rtol=1e-3,
        msg=f"Numerical mismatch for f32 dot (zero acc) with M={M}, N={N}, K={K}",
    )


# ---------------------------------------------------------------------------
# Property Test: Result Type Preservation (Property 2)
# ---------------------------------------------------------------------------

import subprocess
import re
import tempfile
from pathlib import Path


# Type combinations to test: (input_type, output_type)
_TYPE_COMBINATIONS = [
    ("f32", "f32"),
    ("f16", "f32"),
    ("f16", "f16"),
    ("i8", "i32"),
    ("i8", "f32"),
]


def _get_triton_shared_opt_path():
    """Get the triton-shared-opt binary path from environment."""
    path = os.environ.get("TRITON_SHARED_OPT_PATH")
    if path and os.path.isfile(path):
        return path
    # Fallback: try to find it relative to the workspace
    workspace = Path(__file__).resolve().parents[3]
    candidate = (
        workspace
        / "triton"
        / "build"
        / "cmake.linux-x86_64-cpython-3.10"
        / "third_party"
        / "triton_shared"
        / "tools"
        / "triton-shared-opt"
        / "triton-shared-opt"
    )
    if candidate.is_file():
        return str(candidate)
    raise RuntimeError(
        "Cannot find triton-shared-opt. Set TRITON_SHARED_OPT_PATH or source env.sh."
    )


def _generate_dot_mlir(M: int, N: int, K: int, input_type: str, output_type: str) -> str:
    """Generate MLIR text with a tt.dot operation for the given shape and types."""
    return (
        f"module {{\n"
        f"  tt.func @test_dot("
        f"%a: tensor<{M}x{K}x{input_type}>, "
        f"%b: tensor<{K}x{N}x{input_type}>, "
        f"%c: tensor<{M}x{N}x{output_type}>"
        f") -> tensor<{M}x{N}x{output_type}> {{\n"
        f"    %result = tt.dot %a, %b, %c : "
        f"tensor<{M}x{K}x{input_type}> * tensor<{K}x{N}x{input_type}> "
        f"-> tensor<{M}x{N}x{output_type}>\n"
        f"    tt.return %result : tensor<{M}x{N}x{output_type}>\n"
        f"  }}\n"
        f"}}\n"
    )


def _run_decompose_pass(mlir_text: str) -> str:
    """Run the decompose-high-level-ops pass on the given MLIR text and return output IR."""
    opt_path = _get_triton_shared_opt_path()
    result = subprocess.run(
        [opt_path, "--decompose-high-level-ops"],
        input=mlir_text,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"triton-shared-opt failed (rc={result.returncode}):\n"
            f"stdin:\n{mlir_text}\n"
            f"stderr:\n{result.stderr}"
        )
    return result.stdout


def _extract_return_type(output_ir: str) -> str:
    """Extract the return type from the output IR's tt.return statement."""
    # Match the tt.return line and extract the type annotation
    # Pattern: tt.return %... : <type>
    match = re.search(r"tt\.return\s+%\w+\s*:\s*(tensor<[^>]+>)", output_ir)
    if match:
        return match.group(1)
    # Also try matching tt.func return type from the function signature
    # Pattern: ) -> <type> {
    match = re.search(r"\)\s*->\s*(tensor<[^>]+>)\s*\{", output_ir)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract return type from output IR:\n{output_ir}")


# Strategy for type combinations
type_combo_strategy = st.sampled_from(_TYPE_COMBINATIONS)

# Shape strategy using powers of 2 for valid TTIR block sizes.
# Constrained so that M*N*K <= 1048576 (MLIR broadcast element limit).
# Using max 64 for each dimension ensures 64*64*64 = 262144 < 1048576.
pow2_shape_strategy = st.sampled_from([1, 2, 4, 8, 16, 32, 64])


@given(
    M=pow2_shape_strategy,
    N=pow2_shape_strategy,
    K=pow2_shape_strategy,
    type_combo=type_combo_strategy,
)
@settings(
    max_examples=100,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
def test_result_type_preservation(M, N, K, type_combo):
    """Property 2: Result Type Preservation.

    Feature: decompose-high-level-ops, Property 2: Result Type Preservation
    Validates: Requirements 4.4, 8.3

    For any valid tt.dot operation with result type tensor<MxNxR>,
    the final value produced by the decomposed sequence SHALL have exactly
    the same RankedTensorType (shape and element type) as the original DotOp result.
    """
    input_type, output_type = type_combo
    expected_result_type = f"tensor<{M}x{N}x{output_type}>"

    # Generate MLIR with the tt.dot operation
    mlir_text = _generate_dot_mlir(M, N, K, input_type, output_type)

    # Run the decompose pass
    output_ir = _run_decompose_pass(mlir_text)

    # Verify the return type in the decomposed IR matches the expected type
    actual_return_type = _extract_return_type(output_ir)
    assert actual_return_type == expected_result_type, (
        f"Result type mismatch after decompose pass!\n"
        f"  Input type: {input_type}, Output type: {output_type}\n"
        f"  Shape: M={M}, N={N}, K={K}\n"
        f"  Expected: {expected_result_type}\n"
        f"  Actual:   {actual_return_type}\n"
        f"  Output IR:\n{output_ir}"
    )

    # Also verify the function signature return type is preserved
    func_sig_match = re.search(r"\)\s*->\s*(tensor<[^>]+>)\s*\{", output_ir)
    assert func_sig_match is not None, (
        f"Could not find function signature return type in output IR:\n{output_ir}"
    )
    func_return_type = func_sig_match.group(1)
    assert func_return_type == expected_result_type, (
        f"Function signature return type mismatch!\n"
        f"  Expected: {expected_result_type}\n"
        f"  Actual:   {func_return_type}\n"
        f"  Output IR:\n{output_ir}"
    )
