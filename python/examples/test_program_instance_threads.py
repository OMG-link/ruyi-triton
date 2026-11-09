"""Several threads per program instance (num_warps) on the CPU backend.

A loop written as ``for i in cpu.parallel(...)`` is executed by the
``num_warps`` threads of the program instance (buddy-mlir cpar dialect, CPU
runtime in backend/runtime).  The runtime records, through
TRITON_CPU_TRACE_FILE, one row per program instance (``<file>``) and one row
per (instance, region, logical thread) (``<file>.threads``) with the OS thread
id and the CPU measured with sched_getcpu() while the thread runs.

The tests check that

* AC1: a program instance really forks num_warps threads, the result is
  correct, and the code after the loop sees everything the threads wrote
  (join); num_warps=0 takes the inference path,
* AC2: the threads of one instance run on CPUs of a single cluster (L2
  domain), measured with sched_getcpu() and with the threads' affinity masks,
* AC4: more warps per instance means fewer concurrent instances, and the
  total number of threads stays bounded by the number of CPUs,
* AC5: the loop runs on num_warps threads while the code around it runs on
  one thread,
* temporaries of the loop body are private to each thread, and the loop gets
  faster with more threads.
"""

import ctypes
import os
import time
from pathlib import Path

import pytest
import torch

import triton
import triton.language as tl
from triton.language.extra import cpu

from triton.backends.triton_shared import compiler as cpu_compiler
from triton.backends.triton_shared import driver as cpu_driver
from triton.backends.triton_shared.driver import CPUDriver

NUM_THREADS_ENV = "TRITON_CPU_NUM_THREADS"
TRACE_FILE_ENV = "TRITON_CPU_TRACE_FILE"


@pytest.fixture
def backend(monkeypatch):
    triton.runtime.driver.set_active(CPUDriver())
    monkeypatch.delenv(NUM_THREADS_ENV, raising=False)
    monkeypatch.delenv(TRACE_FILE_ENV, raising=False)
    return "cpu"


# ------------------------------------------------------------------ helpers


def runtime():
    path, _ = cpu_driver._get_runtime_library(cpu_driver._get_sanitizer_type())
    lib = ctypes.CDLL(path)
    lib.tcrt_num_slots.argtypes = [ctypes.c_int]
    lib.tcrt_domain_cpus.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    return lib


def runtime_domains():
    lib = runtime()
    domains = []
    for d in range(lib.tcrt_num_domains()):
        buf = (ctypes.c_int * 1024)()
        n = lib.tcrt_domain_cpus(d, buf, 1024)
        domains.append(set(buf[:n]))
    return domains


def sysfs_l2_domain(cpu_id):
    """CPUs sharing the L2 of `cpu_id`, read independently of the runtime."""
    base = Path(f"/sys/devices/system/cpu/cpu{cpu_id}")
    for index in sorted(base.glob("cache/index*")):
        if (index / "level").read_text().strip() == "2":
            text = (index / "shared_cpu_list").read_text().strip()
            cpus = set()
            for item in text.split(","):
                first, _, last = item.partition("-")
                cpus.update(range(int(first), int(last or first) + 1))
            return cpus
    return None


def max_slot_size(warps):
    return max(min(warps, len(d)) for d in runtime_domains())


def read_csv(path):
    lines = Path(path).read_text().splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, map(int, line.split(",")))) for line in lines[1:] if line]


def read_traces(path):
    return read_csv(path), read_csv(str(path) + ".threads")


def threads_by_instance(region_rows):
    out = {}
    for row in region_rows:
        out.setdefault(row["instance"], []).append(row)
    return out


def max_concurrency(instance_rows):
    best = 0
    for a in instance_rows:
        best = max(best, sum(b["start_ns"] <= a["start_ns"] < b["end_ns"] for b in instance_rows))
    return best


def os_thread_count():
    return len(os.listdir("/proc/self/task"))


# ------------------------------------------------------------------ kernels


@triton.jit
def scale_kernel(x_ptr, out_ptr, sum_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    """out = 2 * x + pid, computed by the threads of the instance; then the
    instance's single thread sums what all threads wrote (join check)."""
    pid = tl.program_id(0)
    base = pid * N
    for i in cpu.parallel(0, N, BLOCK):
        offs = base + i + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) * 2.0 + pid)
    # Serial part: reads the values written by every thread of the region.
    total = tl.sum(tl.load(out_ptr + base + tl.arange(0, N)))
    tl.store(sum_ptr + pid, total)


@triton.jit
def busy_kernel(out_ptr, iterations, CHUNKS: tl.constexpr, BLOCK: tl.constexpr):
    """CHUNKS independent dependency-bound chunks per instance."""
    pid = tl.program_id(0)
    for c in cpu.parallel(0, CHUNKS):
        offs = tl.arange(0, BLOCK)
        chunk = pid * CHUNKS + c
        acc = (offs + chunk * BLOCK).to(tl.float32)
        for i in range(iterations):
            acc = acc * 1.0000001 + 1.0
        tl.store(out_ptr + chunk * BLOCK + offs, acc)


@triton.jit
def serial_kernel(out_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    tl.store(out_ptr + offs, offs.to(tl.float32))


def run_scale(num_warps, grid, N=1024, BLOCK=64):
    x = torch.arange(grid * N, dtype=torch.float32) * 0.5
    out = torch.full_like(x, -1.0)
    sums = torch.zeros(grid, dtype=torch.float32)
    kernel = scale_kernel[(grid,)](x, out, sums, N=N, BLOCK=BLOCK, num_warps=num_warps)
    pid = torch.arange(grid, dtype=torch.float32).repeat_interleave(N)
    expected = x * 2.0 + pid
    torch.testing.assert_close(out, expected)
    torch.testing.assert_close(sums, expected.view(grid, N).sum(dim=1))
    return kernel


def calibrate(target_seconds):
    out = torch.zeros(8 * 64, dtype=torch.float32)
    busy_kernel[(1,)](out, 1, CHUNKS=8, BLOCK=64, num_warps=1)
    iterations = 2048
    for _ in range(3):
        start = time.perf_counter()
        busy_kernel[(1,)](out, iterations, CHUNKS=8, BLOCK=64, num_warps=1)
        elapsed = time.perf_counter() - start
        iterations = max(1, int(iterations * target_seconds / max(elapsed, 1e-6)))
    return iterations


# -------------------------------------------------------------------- tests


def test_num_warps_forks_threads(backend, monkeypatch, tmp_path):
    """AC1: num_warps=N forks N threads per instance and joins them."""
    slot = max_slot_size(64)
    if slot < 2:
        pytest.skip("needs a cluster with at least 2 CPUs")
    for warps in sorted({2, slot}):
        trace = tmp_path / f"trace_{warps}.csv"
        monkeypatch.setenv(TRACE_FILE_ENV, str(trace))
        grid = 3
        kernel = run_scale(warps, grid)
        assert kernel.metadata.num_warps == warps
        assert kernel.metadata.cpu_threads == warps

        instances, regions = read_traces(trace)
        assert len(instances) == grid
        per_instance = threads_by_instance(regions)
        assert sorted(per_instance) == list(range(grid))
        for inst, rows in per_instance.items():
            assert sorted(r["tid"] for r in rows) == list(range(warps)), rows
            assert {r["nthreads"] for r in rows} == {warps}
            os_threads = {r["os_tid"] for r in rows}
            assert len(os_threads) == warps, f"instance {inst}: {os_threads}"
            # tid 0 runs on the thread that executes the instance.
            leader = instances[inst]["os_tid"]
            assert next(r["os_tid"] for r in rows if r["tid"] == 0) == leader
            # Join: every thread finished before the instance did.
            assert max(r["end_ns"] for r in rows) <= instances[inst]["end_ns"]
        print(f"num_warps={warps}: {grid} instances x {warps} OS threads")


def test_num_warps_zero_infers(backend, monkeypatch, tmp_path):
    """AC1: num_warps=0 resolves to the inferred thread count."""
    inferred = cpu_compiler._infer_num_warps()
    trace = tmp_path / "trace_infer.csv"
    monkeypatch.setenv(TRACE_FILE_ENV, str(trace))
    kernel = run_scale(0, 2)
    assert kernel.metadata.num_warps == inferred >= 1
    _, regions = read_traces(trace)
    for rows in threads_by_instance(regions).values():
        assert {r["nthreads"] for r in rows} == {inferred}
        assert len({r["os_tid"] for r in rows}) == min(inferred, max_slot_size(inferred))
    print(f"num_warps=0 -> {inferred} threads per instance")


def test_threads_stay_in_one_cluster(backend, monkeypatch, tmp_path):
    """AC2: measured CPUs and affinity of an instance's threads lie in one L2 domain."""
    slot = max_slot_size(64)
    if slot < 2:
        pytest.skip("needs a cluster with at least 2 CPUs")
    trace = tmp_path / "trace_cluster.csv"
    monkeypatch.setenv(TRACE_FILE_ENV, str(trace))
    iterations = calibrate(0.002)
    grid = 2 * runtime().tcrt_num_slots(slot)
    out = torch.zeros(grid * 8 * 64, dtype=torch.float32)
    busy_kernel[(grid,)](out, iterations, CHUNKS=8, BLOCK=64, num_warps=slot)

    _, regions = read_traces(trace)
    if any(r["cpu"] < 0 for r in regions):
        pytest.skip("sched_getcpu() is not available")
    domains = runtime_domains()
    for inst, rows in threads_by_instance(regions).items():
        cpus = {r["cpu"] for r in rows}
        assert sum(cpus <= d for d in domains) == 1, f"instance {inst} on CPUs {cpus}"
        # Independent of the runtime's own topology detection.
        l2 = sysfs_l2_domain(min(cpus))
        if l2 is not None:
            assert cpus <= l2, f"instance {inst}: CPUs {cpus} not in L2 domain {l2}"
        # The threads cannot migrate: their affinity is a single CPU of the domain.
        for os_tid in {r["os_tid"] for r in rows}:
            allowed = os.sched_getaffinity(os_tid)
            assert len(allowed) == 1, f"thread {os_tid} may run on {allowed}"
            if l2 is not None:
                assert allowed <= l2, f"thread {os_tid} pinned outside {l2}: {allowed}"
    print(f"{grid} instances x {slot} threads, each instance inside one L2 domain")


@triton.jit
def placement_kernel(cpu_ptr, tid_ptr, work_ptr, ITERS: tl.constexpr, SPIN: tl.constexpr):
    """Every iteration stores the CPU / thread id of the thread running it."""
    pid = tl.program_id(0)
    for i in cpu.parallel(0, ITERS):
        offs = tl.arange(0, 16)
        acc = (offs + i).to(tl.float32)
        for _ in range(SPIN):
            acc = acc * 1.0000001 + 1.0
        tl.store(work_ptr + (pid * ITERS + i) * 16 + offs, acc)
        tl.store(cpu_ptr + pid * ITERS + i, cpu.current_cpu())
        tl.store(tid_ptr + pid * ITERS + i, cpu.os_thread_id())


def test_kernel_reports_one_cluster(backend):
    """AC2, measured by the kernel itself: cpu.current_cpu() in every iteration."""
    slot = max_slot_size(64)
    if slot < 2:
        pytest.skip("needs a cluster with at least 2 CPUs")
    grid, iters = 2 * runtime().tcrt_num_slots(slot), 2 * slot
    cpus = torch.full((grid * iters,), -1, dtype=torch.int32)
    tids = torch.full((grid * iters,), -1, dtype=torch.int32)
    work = torch.zeros((grid * iters * 16,), dtype=torch.float32)
    placement_kernel[(grid,)](cpus, tids, work, ITERS=iters, SPIN=20000, num_warps=slot)
    cpus, tids = cpus.view(grid, iters).tolist(), tids.view(grid, iters).tolist()
    if any(c < 0 for row in cpus for c in row):
        pytest.skip("sched_getcpu() is not available")
    for pid in range(grid):
        used = set(cpus[pid])
        l2 = sysfs_l2_domain(min(used))
        assert l2 is None or used <= l2, f"instance {pid}: CPUs {used}, L2 domain {l2}"
        assert len(set(tids[pid])) == slot, f"instance {pid}: threads {set(tids[pid])}"
        # Round robin: iteration i is run by thread i % slot, always on the same CPU.
        for i in range(iters):
            assert tids[pid][i] == tids[pid][i % slot] and cpus[pid][i] == cpus[pid][i % slot]
        print(f"instance {pid}: {slot} threads on CPUs {sorted(used)} (L2 {sorted(l2 or [])})")


def test_warps_reduce_instance_parallelism(backend, monkeypatch, tmp_path):
    """AC4: concurrent instances drop as num_warps grows; threads stay bounded."""
    lib = runtime()
    # The pool's CPUs; on the K3 these are the AI cores, outside the affinity
    # of this thread.
    ncpu = lib.tcrt_num_cpus()
    if ncpu < 2:
        pytest.skip("needs at least 2 CPUs")
    iterations = calibrate(0.004)
    warps_list = sorted({1, 2, max_slot_size(64)})
    concurrency = {}
    baseline_threads = None
    for warps in warps_list:
        trace = tmp_path / f"trace_w{warps}.csv"
        monkeypatch.setenv(TRACE_FILE_ENV, str(trace))
        grid = 2 * ncpu
        out = torch.zeros(grid * 8 * 64, dtype=torch.float32)
        busy_kernel[(grid,)](out, iterations, CHUNKS=8, BLOCK=64, num_warps=warps)
        instances, regions = read_traces(trace)
        slots = lib.tcrt_num_slots(warps)
        concurrency[warps] = max_concurrency(instances)
        assert concurrency[warps] <= slots, (warps, concurrency[warps], slots)
        # One instance per slot at a time: at most `slots` leaders.
        assert len({r["thread_id"] for r in instances}) <= slots
        used = {r["os_tid"] for r in instances} | {r["os_tid"] for r in regions}
        assert len(used) <= ncpu, f"{len(used)} threads used for {ncpu} CPUs"
        threads_now = os_thread_count()
        baseline_threads = baseline_threads or threads_now
        # The pool is created once; larger num_warps never adds threads.
        assert threads_now == baseline_threads, (warps, threads_now, baseline_threads)
        print(f"num_warps={warps}: slots={slots} max concurrent instances="
              f"{concurrency[warps]} threads used={len(used)} process threads={threads_now}")
    values = [concurrency[w] for w in warps_list]
    assert values == sorted(values, reverse=True) and values[0] > values[-1], concurrency


def test_parallel_and_serial_parts(backend, monkeypatch, tmp_path):
    """AC5: the cpu.parallel loop runs on num_warps threads, the rest on one."""
    slot = max_slot_size(64)
    if slot < 2:
        pytest.skip("needs a cluster with at least 2 CPUs")
    trace = tmp_path / "trace_parts.csv"
    monkeypatch.setenv(TRACE_FILE_ENV, str(trace))
    grid = 2
    run_scale(slot, grid)
    instances, regions = read_traces(trace)
    per_instance = threads_by_instance(regions)
    for inst in range(grid):
        region_threads = {r["os_tid"] for r in per_instance[inst]}
        # The serial part (everything outside the loop, including the final
        # reduction) runs on the instance's thread only: it is the thread
        # recorded for the instance, and no other thread runs outside regions.
        serial_threads = {instances[inst]["os_tid"]}
        assert len(region_threads) == slot and len(serial_threads) == 1
        assert serial_threads < region_threads
        print(f"instance {inst}: parallel part {len(region_threads)} threads, "
              f"serial part {len(serial_threads)} thread")

    # A kernel without cpu.parallel reserves no extra cores (cpu_threads=1),
    # whatever num_warps says.
    out = torch.zeros(4 * 64, dtype=torch.float32)
    kernel = serial_kernel[(4,)](out, BLOCK=64, num_warps=slot)
    assert kernel.metadata.num_warps == slot and kernel.metadata.cpu_threads == 1
    _, regions = read_traces(trace)
    assert regions == []


def test_threads_do_not_share_temporaries(backend):
    """Tensors computed inside the loop are private to each thread.

    In busy_kernel the accumulator of every chunk is a temporary tensor.  If
    the threads of an instance shared its buffer, they would race on it and
    the result would differ from the single thread run.
    """
    slot = max_slot_size(64)
    if slot < 2:
        pytest.skip("needs a cluster with at least 2 CPUs")
    grid, iterations = 3, 2000
    expected = torch.zeros(grid * 8 * 64, dtype=torch.float32)
    busy_kernel[(grid,)](expected, iterations, CHUNKS=8, BLOCK=64, num_warps=1)
    for warps in sorted({2, slot}):
        for _ in range(3):
            out = torch.zeros_like(expected)
            busy_kernel[(grid,)](out, iterations, CHUNKS=8, BLOCK=64, num_warps=warps)
            torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_region_speedup(backend, monkeypatch):
    """The parallel loop of a single instance gets faster with more threads."""
    slot = max_slot_size(64)
    if slot < 2:
        pytest.skip("needs a cluster with at least 2 CPUs")
    iterations = calibrate(0.05)
    out = torch.zeros(8 * 64, dtype=torch.float32)

    def measure(warps):
        busy_kernel[(1,)](out, iterations, CHUNKS=8, BLOCK=64, num_warps=warps)
        best = float("inf")
        for _ in range(3):
            start = time.perf_counter()
            busy_kernel[(1,)](out, iterations, CHUNKS=8, BLOCK=64, num_warps=warps)
            best = min(best, time.perf_counter() - start)
        return best

    single, multi = measure(1), measure(slot)
    print(f"one instance: num_warps=1 {single * 1e3:.1f}ms, num_warps={slot} "
          f"{multi * 1e3:.1f}ms, speedup {single / multi:.2f}x")
    assert single / multi > 1.3
