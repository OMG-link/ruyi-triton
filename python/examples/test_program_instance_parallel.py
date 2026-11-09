"""Program instance (launch grid) level parallelism of the CPU backend.

The CPU runtime (backend/runtime/triton_cpu_rt.*) executes the launch grid on
a pool of worker threads, one per available CPU and pinned to it, and hands
program instances out dynamically.  These tests check that

* every program instance runs exactly once and receives the right program ids,
* instances really overlap in time (recorded by the runtime's trace facility
  through TRITON_CPU_TRACE_FILE, no kernel instrumentation needed),
* every worker stays on its CPU,
* TRITON_CPU_NUM_THREADS=1 gives a true serial baseline,
* more workers actually reduce wall-clock time.

As a serial execution can never interleave two program instances, an observed
interleaving is proof that the grid is no longer traversed serially.
"""

import ctypes
import os
import time

import pytest
import torch

import triton
import triton.language as tl

from triton.backends.triton_shared import driver as cpu_driver
from triton.backends.triton_shared.driver import CPUDriver

NUM_THREADS_ENV = "TRITON_CPU_NUM_THREADS"
TRACE_FILE_ENV = "TRITON_CPU_TRACE_FILE"
TRACE_HEADER = "instance,start_ns,end_ns,thread_id,cpu,os_tid"

# Elements per program instance in the busy kernel; kept small so that the
# calibration below can tune the amount of work per instance.
BUSY_BLOCK = 64


def available_cpus():
    """CPUs of the runtime's worker pool.

    Not os.sched_getaffinity(0): on the SpacemiT K3 the pool runs on the AI
    cores, which are outside the affinity of the calling thread.
    """
    path, _ = cpu_driver._get_runtime_library(cpu_driver._get_sanitizer_type())
    return ctypes.CDLL(path).tcrt_num_cpus()


@pytest.fixture
def backend():
    triton.runtime.driver.set_active(CPUDriver())
    return "cpu"


@triton.jit
def instance_id_kernel(out_ptr, ny: tl.constexpr, nz: tl.constexpr):
    """Store the flat instance id at out[flat_id].

    The flat id follows the launcher's traversal order (x outer, z inner), so
    the test also covers the grid <-> program id mapping of the launcher.
    """
    pid_x = tl.program_id(0)
    pid_y = tl.program_id(1)
    pid_z = tl.program_id(2)
    instance_id = (pid_x * ny + pid_y) * nz + pid_z
    tl.store(out_ptr + instance_id, instance_id)


@triton.jit
def busy_kernel(out_ptr, iterations, BLOCK: tl.constexpr):
    """Dependency-bound loop; every iteration does BLOCK float operations.

    `iterations` is a runtime argument (not constexpr) so the loop cannot be
    unrolled or folded at compile time.
    """
    pid = tl.program_id(0)
    offsets = tl.arange(0, BLOCK)
    acc = (offsets + pid * BLOCK).to(tl.float32)
    for i in range(iterations):
        acc = acc * 1.0000001 + 1.0
    tl.store(out_ptr + pid * BLOCK + offsets, acc)


def read_trace(path):
    rows = []
    with open(path) as trace:
        header = trace.readline().strip()
        assert header == TRACE_HEADER, f"unexpected trace header: {header!r}"
        for line in trace:
            line = line.strip()
            if not line:
                continue
            instance, start, end, thread_id, cpu, _os_tid = line.split(",")
            rows.append((int(instance), int(start), int(end), int(thread_id), int(cpu)))
    return rows


def interleaved_pairs(rows):
    """Pairs (a, b) with a.start < b.start < a.end < b.end.

    A serial traversal yields disjoint intervals, hence no such pair.
    """
    return [(a, b) for a in rows for b in rows if a[1] < b[1] < a[2] < b[2]]


def run_busy_kernel(backend, num_instances, iterations):
    out = torch.zeros(
        (num_instances * BUSY_BLOCK,), dtype=torch.float32, device=backend
    )
    busy_kernel[(num_instances,)](out, iterations, BLOCK=BUSY_BLOCK)
    return out


def calibrate_iterations(backend, target_seconds):
    """Iteration count that keeps one program instance busy for ~target_seconds.

    Every launch pays a fixed overhead (the launcher extension is reloaded for
    each launch), so the iteration count is refined a few times from a measured
    launch instead of being computed once from scratch.
    """
    out = torch.zeros((BUSY_BLOCK,), dtype=torch.float32, device=backend)
    # Warm up: the first launch compiles and caches the launcher.
    busy_kernel[(1,)](out, 1, BLOCK=BUSY_BLOCK)

    def time_one(iterations):
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            busy_kernel[(1,)](out, iterations, BLOCK=BUSY_BLOCK)
            best = min(best, time.perf_counter() - start)
        return best

    iterations = 4096
    for _ in range(4):
        elapsed = time_one(iterations)
        if elapsed <= 0.0:
            break
        iterations = max(1, min(int(iterations * target_seconds / elapsed), 1 << 30))
    return iterations


def test_grid_parallel_coverage(backend, monkeypatch):
    """Every instance runs exactly once, with the program ids of its grid slot."""
    monkeypatch.delenv(NUM_THREADS_ENV, raising=False)
    monkeypatch.delenv(TRACE_FILE_ENV, raising=False)

    for grid in [(16, 1, 1), (2, 3, 4)]:
        ny, nz = grid[1], grid[2]
        num_instances = grid[0] * ny * nz
        out = torch.full((num_instances,), -1, dtype=torch.int32, device=backend)
        instance_id_kernel[grid](out, ny=ny, nz=nz)
        torch.testing.assert_close(
            out, torch.arange(num_instances, dtype=torch.int32, device=backend)
        )


def test_grid_parallel_overlap(backend, monkeypatch, tmp_path):
    """Program instances overlap in time when the grid runs in parallel."""
    cpus = available_cpus()
    if cpus < 2:
        pytest.skip("program instance parallelism needs at least 2 CPUs")

    monkeypatch.delenv(NUM_THREADS_ENV, raising=False)
    trace_path = tmp_path / "trace.csv"
    monkeypatch.setenv(TRACE_FILE_ENV, str(trace_path))

    num_instances = max(2 * cpus, 8)
    iterations = calibrate_iterations(backend, target_seconds=0.005)
    out = run_busy_kernel(backend, num_instances, iterations)

    rows = read_trace(trace_path)
    assert [row[0] for row in rows] == list(range(num_instances)), rows
    assert torch.isfinite(out).all()

    thread_ids = sorted({row[3] for row in rows})
    assert len(thread_ids) >= 2, f"only one launcher thread: {thread_ids}"

    pairs = interleaved_pairs(rows)
    assert pairs, "no two program instances overlapped; the grid ran serially"

    busy = sum(row[2] - row[1] for row in rows)
    span = max(row[2] for row in rows) - min(row[1] for row in rows)
    concurrency = busy / span
    assert concurrency > 1.5, f"achieved concurrency {concurrency:.2f}"


def test_grid_workers_pinned(backend, monkeypatch, tmp_path):
    """Each worker runs all of its instances on one CPU, workers on distinct CPUs."""
    cpus = available_cpus()
    if cpus < 2:
        pytest.skip("program instance parallelism needs at least 2 CPUs")

    monkeypatch.delenv(NUM_THREADS_ENV, raising=False)
    trace_path = tmp_path / "trace_pinned.csv"
    monkeypatch.setenv(TRACE_FILE_ENV, str(trace_path))

    num_instances = 4 * cpus
    iterations = calibrate_iterations(backend, target_seconds=0.002)
    run_busy_kernel(backend, num_instances, iterations)

    rows = read_trace(trace_path)
    if any(row[4] < 0 for row in rows):
        pytest.skip("the runtime cannot report the current CPU on this host")
    cpus_of_worker = {}
    for row in rows:
        cpus_of_worker.setdefault(row[3], set()).add(row[4])
    assert len(cpus_of_worker) >= 2, cpus_of_worker
    for worker, worker_cpus in cpus_of_worker.items():
        assert len(worker_cpus) == 1, f"worker {worker} migrated: {sorted(worker_cpus)}"
    used = [next(iter(c)) for c in cpus_of_worker.values()]
    assert len(set(used)) == len(used), f"two workers share a CPU: {cpus_of_worker}"


def test_grid_parallel_serial_override(backend, monkeypatch, tmp_path):
    """TRITON_CPU_NUM_THREADS=1 restores the serial traversal."""
    monkeypatch.setenv(NUM_THREADS_ENV, "1")
    trace_path = tmp_path / "trace_serial.csv"
    monkeypatch.setenv(TRACE_FILE_ENV, str(trace_path))

    num_instances = 8
    out = torch.full((num_instances,), -1, dtype=torch.int32, device=backend)
    instance_id_kernel[(num_instances, 1, 1)](out, ny=1, nz=1)

    rows = read_trace(trace_path)
    assert {row[3] for row in rows} == {0}, rows
    assert not interleaved_pairs(rows), "serial baseline interleaved instances"
    torch.testing.assert_close(
        out, torch.arange(num_instances, dtype=torch.int32, device=backend)
    )


def test_grid_parallel_speedup(backend, monkeypatch):
    """More workers make the grid finish faster."""
    cpus = available_cpus()
    if cpus < 2:
        pytest.skip("program instance parallelism needs at least 2 CPUs")

    monkeypatch.delenv(TRACE_FILE_ENV, raising=False)
    num_instances = max(2 * cpus, 8)
    iterations = calibrate_iterations(backend, target_seconds=0.02)
    out = torch.zeros(
        (num_instances * BUSY_BLOCK,), dtype=torch.float32, device=backend
    )

    def measure(threads):
        monkeypatch.setenv(NUM_THREADS_ENV, str(threads))
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            busy_kernel[(num_instances,)](out, iterations, BLOCK=BUSY_BLOCK)
            best = min(best, time.perf_counter() - start)
        return best

    serial = measure(1)
    parallel = measure(cpus)
    speedup = serial / parallel
    print(
        f"serial={serial * 1e3:.2f}ms parallel={parallel * 1e3:.2f}ms "
        f"speedup={speedup:.2f}x ({cpus} cpus)"
    )
    # The overlap test is the authoritative check; here a conservative bound
    # keeps the test meaningful on small/loaded CI machines.
    assert speedup > 1.3, f"only {speedup:.2f}x speedup with {cpus} cpus"
