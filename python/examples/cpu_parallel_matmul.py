"""Runnable example: matrix multiplication with cpu.parallel.

    source env.sh
    python triton-riscv/python/examples/cpu_parallel_matmul.py [M N K]

Each program instance computes a (SPLIT_M * BLOCK_M) x BLOCK_N tile of
C = A @ B.  The tile is cut into SPLIT_M row blocks of BLOCK_M rows, and the
`cpu.parallel` loop hands the row blocks to the num_warps threads of the
instance: thread t computes the row blocks t, t + num_warps, ...  Every row
block runs its own K loop with a private accumulator and writes a disjoint part
of C, so the iterations are independent.

With a small grid (fewer instances than cores) the instances alone cannot
keep all cores busy; the threads of each instance do.  The script runs the
same kernel with num_warps=1 (row blocks one after the other) and with the
inferred num_warps (one cluster per instance), checks both against torch, and
prints the times and the CPUs that ran each instance (runtime trace).
"""

import os
import sys
import tempfile
import time
from pathlib import Path

import torch

import triton
import triton.language as tl
from triton.language.extra import cpu
from triton.backends.triton_shared.driver import CPUDriver


@triton.jit
def matmul_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    SPLIT_M: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    # Shared by all threads of the instance: defined before the loop.
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    # One row block per iteration, executed by the num_warps threads.
    for s in cpu.parallel(0, SPLIT_M):
        offs_m = (pid_m * SPLIT_M + s) * BLOCK_M + tl.arange(0, BLOCK_M)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        # Private to the thread running this iteration.
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        # Ordinary loop inside the parallel one: sequential in each thread.
        for k in range(0, tl.cdiv(K, BLOCK_K)):
            k_mask = offs_k < K - k * BLOCK_K
            a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & k_mask[None, :], other=0.0)
            b = tl.load(b_ptrs, mask=k_mask[:, None] & (offs_n[None, :] < N), other=0.0)
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


BLOCK_M, BLOCK_N, BLOCK_K, SPLIT_M = 32, 64, 16, 8


def matmul(a, b, num_warps):
    M, K = a.shape
    K2, N = b.shape
    assert K == K2, "incompatible dimensions"
    c = torch.empty((M, N), dtype=torch.float32)
    grid = (triton.cdiv(M, BLOCK_M * SPLIT_M), triton.cdiv(N, BLOCK_N))
    kernel = matmul_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_M=SPLIT_M,
        num_warps=num_warps,
    )
    return c, kernel, grid


def best_time(fn, repeat=5):
    fn()  # compile / warm up
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def placement(trace):
    """instance -> (OS threads, CPUs, start ms, end ms), times relative to the launch.

    Instances are handed out dynamically: a cluster that finishes its
    instance takes the next one.  If one cluster is slowed down (e.g. by
    another program on the same cores), the other one executes more
    instances; the start/end times show it.
    """
    instances = [list(map(int, line.split(",")))
                 for line in Path(trace).read_text().splitlines()[1:] if line]
    t0 = min(row[1] for row in instances)
    out = {row[0]: (set(), set(), (row[1] - t0) / 1e6, (row[2] - t0) / 1e6) for row in instances}
    for line in Path(f"{trace}.threads").read_text().splitlines()[1:]:
        instance, _region, _tid, _n, os_tid, cpu_id, _start, _end = map(int, line.split(","))
        out[instance][0].add(os_tid)
        out[instance][1].add(cpu_id)
    return out


def main():
    M, N, K = (int(v) for v in sys.argv[1:4]) if len(sys.argv) >= 4 else (512, 128, 512)
    triton.runtime.driver.set_active(CPUDriver())
    torch.manual_seed(0)
    a = torch.randn((M, K), dtype=torch.float32)
    b = torch.randn((K, N), dtype=torch.float32)
    expected = torch.matmul(a, b)

    results = {}
    for num_warps in (1, 0):  # 0: inferred (one cluster per instance)
        c, kernel, grid = matmul(a, b, num_warps)
        torch.testing.assert_close(c, expected, atol=1e-2, rtol=1e-4)
        seconds = best_time(lambda: matmul(a, b, num_warps))
        results[kernel.metadata.num_warps] = seconds
        print(f"num_warps={num_warps} -> {kernel.metadata.num_warps} threads per instance, "
              f"grid {grid[0]}x{grid[1]}: {seconds * 1e3:.1f} ms, result OK")

    trace = Path(tempfile.mkdtemp()) / "trace.csv"
    os.environ["TRITON_CPU_TRACE_FILE"] = str(trace)
    _, kernel, _ = matmul(a, b, 0)
    del os.environ["TRITON_CPU_TRACE_FILE"]
    for instance, (threads, cpus, start, end) in sorted(placement(trace).items()):
        print(f"  instance {instance}: {SPLIT_M} row blocks on {len(threads)} OS threads, "
              f"CPUs {sorted(cpus)}, {start:6.1f} -> {end:6.1f} ms ({end - start:.1f} ms)")

    warps = max(results)
    print(f"M={M} N={N} K={K}: num_warps={warps} is {results[1] / results[warps]:.2f}x "
          f"faster than num_warps=1")


if __name__ == "__main__":
    main()
