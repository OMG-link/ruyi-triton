"""Program instance parallelism and threads per program instance.

The CPU runtime (backend/runtime) executes the launch grid on a pool of
pinned workers, one per compute CPU of the board configuration, and
tl.dev.parallel loops on the threads of one program instance's slot. See
docs/09-Program-Instance-Parallelism.md and docs/10-Program-Instance-Threads.md.

Runtime state is per process and configured by the first launch, so the
tests that need a different configuration (TRITON_CPAR_CPUS) run in a
subprocess.
"""

import csv
import json
import os
import subprocess
import sys
import threading

import pytest
import torch

import triton
import triton.language as tl


def _read_trace(path):
    with open(path) as f:
        instances = list(csv.DictReader(f))
    with open(f"{path}.threads") as f:
        threads = list(csv.DictReader(f))
    return instances, threads


def _max_concurrency(rows):
    events = sorted([(int(r["start_ns"]), 1) for r in rows] +
                    [(int(r["end_ns"]), -1) for r in rows])
    live = best = 0
    for _, delta in events:
        live += delta
        best = max(best, live)
    return best


def _run(code, tmp_path, env=None):
    # @triton.jit needs the kernel's source in a file.
    script = tmp_path / "child.py"
    script.write_text(code)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True,
                            env={**os.environ, **(env or {})}, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


# --------------------------------------------------------------------------
# Launch grid
# --------------------------------------------------------------------------


@triton.jit
def _ids_kernel(out_ptr, BUSY: tl.constexpr):
    x = tl.program_id(0)
    y = tl.program_id(1)
    z = tl.program_id(2)
    acc = 0.0
    for i in range(BUSY):
        acc += i * 1e-9
    linear = (x * tl.num_programs(1) + y) * tl.num_programs(2) + z
    tl.store(out_ptr + linear, linear + acc.to(tl.int32) * 0)


@pytest.mark.parametrize("grid", [(1, 1, 1), (37, 1, 1), (3, 4, 5)])
def test_grid_coverage(device, grid):
    total = grid[0] * grid[1] * grid[2]
    out = torch.full((total,), -1, dtype=torch.int32)
    _ids_kernel[grid](out, BUSY=1)
    torch.testing.assert_close(out, torch.arange(total, dtype=torch.int32))


def test_instances_overlap(device, tmp_path):
    if len(os.sched_getaffinity(0)) < 2:
        pytest.skip("needs two CPUs")
    trace = tmp_path / "trace.csv"
    out = torch.empty(64, dtype=torch.int32)
    os.environ["TRITON_CPAR_TRACE_FILE"] = str(trace)
    try:
        _ids_kernel[(64,)](out, BUSY=200000)
    finally:
        del os.environ["TRITON_CPAR_TRACE_FILE"]
    instances, threads = _read_trace(trace)
    assert sorted(int(r["instance"]) for r in instances) == list(range(64))
    # A serial traversal never produces interleaved instance intervals.
    assert _max_concurrency(instances) > 1
    # No parallel region: no region rows.
    assert threads == []


_SERIAL = """
import os, csv, torch, triton, triton.language as tl
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())

@triton.jit
def k(out_ptr):
    pid = tl.program_id(0)
    tl.store(out_ptr + pid, pid)

out = torch.empty(16, dtype=torch.int32)
k[(16,)](out)
assert out.tolist() == list(range(16))
with open(os.environ["TRITON_CPAR_TRACE_FILE"]) as f:
    rows = list(csv.DictReader(f))
print(sorted({r["cpu"] for r in rows}), len({r["os_tid"] for r in rows}))
"""


def test_single_cpu_override(device, tmp_path):
    cpu = sorted(os.sched_getaffinity(0))[-1]
    out = _run(_SERIAL, tmp_path, {"TRITON_CPAR_CPUS": str(cpu),
                                   "TRITON_CPAR_TRACE_FILE": str(tmp_path / "t.csv")})
    assert out.split() == [f"['{cpu}']", "1"], out


# --------------------------------------------------------------------------
# Slot layout
# --------------------------------------------------------------------------

_LAYOUT = """
import json
from triton.backends.triton_shared import board, cpu_runtime
cpu_runtime.configure(board.GENERIC)
print(json.dumps({w: [cpu_runtime.slot_cpus(w, s) for s in range(cpu_runtime.num_slots(w))]
                  for w in (1, 2, 3, 4, 6, 8, 16)}))
"""


def test_slot_layout(device):
    if len(os.sched_getaffinity(0)) < 8:
        pytest.skip("needs eight CPUs")
    cpus = sorted(os.sched_getaffinity(0))[:8]
    a, b = cpus[:4], cpus[4:]
    spec = ",".join(map(str, a)) + ";" + ",".join(map(str, b))
    result = subprocess.run([sys.executable, "-c", _LAYOUT], capture_output=True, text=True,
                            env={**os.environ, "TRITON_CPAR_CPUS": spec}, timeout=600)
    assert result.returncode == 0, result.stderr
    layouts = {int(w): slots for w, slots in json.loads(result.stdout).items()}
    # Round robin over the domains, in launch order.
    assert sorted(map(tuple, layouts[1])) == [(c,) for c in cpus]
    assert layouts[1][:2] == [[a[0]], [b[0]]]
    assert sorted(layouts[2]) == sorted([a[:2], a[2:], b[:2], b[2:]])
    # The rest of a domain too small for another slot stays idle.
    assert sorted(layouts[3]) == sorted([a[:3], b[:3]])
    assert sorted(layouts[4]) == sorted([a, b])
    # A slot larger than a domain starts at a domain boundary and spans the
    # next one (with a warning); what is left is too small for another.
    assert layouts[6] == [cpus[:6]]
    assert "span several domains" in result.stderr
    assert layouts[8] == [cpus]
    # Never more CPUs than the configuration.
    assert layouts[16] == [cpus]


# --------------------------------------------------------------------------
# tl.dev.parallel
# --------------------------------------------------------------------------


@triton.jit
def _parallel_kernel(x_ptr, out_ptr, sum_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    for i in tl.dev.parallel(0, N, BLOCK):
        offs = base + i + tl.arange(0, BLOCK)
        v = tl.load(x_ptr + offs)
        # Temporaries of the body belong to one iteration.
        t = v * 2.0 + 1.0
        tl.store(out_ptr + offs, t * t)
    # After the join, one thread reads what all of them wrote.
    tl.store(sum_ptr + pid, tl.sum(tl.load(out_ptr + base + tl.arange(0, N))))


def _parallel(num_warps, grid=3, n=4096, block=128, trace=None):
    torch.manual_seed(0)
    x = torch.randn(grid * n)
    out = torch.empty_like(x)
    sums = torch.empty(grid)
    if trace:
        os.environ["TRITON_CPAR_TRACE_FILE"] = str(trace)
    try:
        compiled = _parallel_kernel[(grid,)](x, out, sums, N=n, BLOCK=block, num_warps=num_warps)
    finally:
        os.environ.pop("TRITON_CPAR_TRACE_FILE", None)
    ref = (x * 2.0 + 1.0) ** 2
    return compiled, x, out, sums, ref


@pytest.mark.parametrize("num_warps", [1, 2, 4])
def test_parallel_results(device, num_warps):
    compiled, x, out, sums, ref = _parallel(num_warps)
    torch.testing.assert_close(out, ref)
    torch.testing.assert_close(sums, ref.reshape(3, -1).sum(1), rtol=1e-4, atol=1e-2)
    assert compiled.metadata.num_warps == num_warps
    # With one thread the loop stays sequential: no region, one CPU.
    assert compiled.metadata.cpu_threads == num_warps


def test_parallel_bit_identical_across_threads(device):
    *_, out1, _, _ = _parallel(1)
    *_, out4, _, _ = _parallel(4)
    assert torch.equal(out1, out4)


def test_parallel_forks_threads(device, tmp_path):
    from triton.backends.triton_shared import cpu_runtime

    trace = tmp_path / "trace.csv"
    compiled, *_ = _parallel(4, trace=trace)
    assert compiled.metadata.cpu_threads == 4
    instances, threads = _read_trace(trace)
    by_instance = {}
    for row in threads:
        by_instance.setdefault(row["instance"], []).append(row)
    assert sorted(by_instance) == ["0", "1", "2"]
    width = len(cpu_runtime.slot_cpus(4, 0))
    for inst, rows in by_instance.items():
        assert sorted(int(r["tid"]) for r in rows) == [0, 1, 2, 3]
        assert {int(r["nthreads"]) for r in rows} == {4}
        # One OS thread per CPU of the slot; logical ids beyond it share them.
        assert len({r["os_tid"] for r in rows}) == width
        # tid 0 runs on the thread that runs the instance.
        leader = next(r for r in instances if r["instance"] == inst)["os_tid"]
        assert next(r for r in rows if r["tid"] == "0")["os_tid"] == leader


def test_kernel_without_parallel_region_keeps_one_cpu(device):
    out = torch.empty(4, dtype=torch.int32)
    compiled = _ids_kernel[(4,)](out, BUSY=1, num_warps=4)
    assert compiled.metadata.num_warps == 4
    assert compiled.metadata.cpu_threads == 1


def test_num_warps_zero_is_one_cluster(device):
    from triton.backends.triton_shared import board

    compiled, *_ = _parallel(0)
    assert compiled.metadata.num_warps == board.cluster_size(compiled.metadata.board_config)


@triton.jit
def _carried_kernel(out_ptr, N: tl.constexpr):
    acc = 0
    for i in tl.dev.parallel(0, N):
        acc += i
    tl.store(out_ptr, acc)


def test_parallel_rejects_carried_values(device):
    out = torch.empty(1, dtype=torch.int32)
    with pytest.raises(Exception, match="must not assign to acc"):
        _carried_kernel[(1,)](out, N=8)


@triton.jit
def _nested_kernel(out_ptr, N: tl.constexpr, M: tl.constexpr):
    for i in tl.dev.parallel(0, N):
        for j in tl.dev.parallel(0, M):
            tl.store(out_ptr + i * M + j, i * M + j)


def test_nested_parallel_loops(device):
    out = torch.full((8 * 16,), -1, dtype=torch.int32)
    _nested_kernel[(1,)](out, N=8, M=16, num_warps=4)
    torch.testing.assert_close(out, torch.arange(8 * 16, dtype=torch.int32))


# --------------------------------------------------------------------------
# tl.dev.current_cpu / tl.dev.os_thread_id
# --------------------------------------------------------------------------


@triton.jit
def _where_kernel(cpu_ptr, tid_ptr, serial_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    for i in tl.dev.parallel(0, N):
        tl.store(cpu_ptr + pid * N + i, tl.dev.current_cpu())
        tl.store(tid_ptr + pid * N + i, tl.dev.os_thread_id())
    tl.store(serial_ptr + pid * 2, tl.dev.current_cpu())
    tl.store(serial_ptr + pid * 2 + 1, tl.dev.os_thread_id())


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
def test_thread_queries(device):
    from triton.backends.triton_shared import cpu_runtime

    grid, n, warps = 3, 16, 4
    cpus = torch.full((grid, n), -2, dtype=torch.int32)
    tids = torch.zeros((grid, n), dtype=torch.int32)
    serial = torch.zeros((grid, 2), dtype=torch.int32)
    _where_kernel[(grid,)](cpus, tids, serial, N=n, num_warps=warps)
    slots = [cpu_runtime.slot_cpus(warps, s) for s in range(cpu_runtime.num_slots(warps))]
    width = len(slots[0])
    for p in range(grid):
        row_cpus, row_tids = cpus[p].tolist(), tids[p].tolist()
        # All iterations of an instance ran on the CPUs of one slot.
        slot = next(s for s in slots if row_cpus[0] in s)
        assert set(row_cpus) <= set(slot)
        # Iteration i runs on thread i % W, which shares an OS thread with
        # i % width when the slot is smaller than W.
        for i in range(n):
            assert row_tids[i] == row_tids[i % warps % width]
            assert row_cpus[i] == row_cpus[i % warps % width]
        assert len(set(row_tids)) == min(width, warps)
        # The code after the loop runs on the thread of logical id 0.
        assert serial[p].tolist() == [row_cpus[0], row_tids[0]]
    # The queries ran on workers, never on the calling thread.
    assert threading.get_native_id() not in tids.flatten().tolist()


# --------------------------------------------------------------------------
# tl.dev.num_threads
# --------------------------------------------------------------------------


@triton.jit
def _per_thread_kernel(out_ptr, threads_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    # One row of scratch per thread, split by thread as an OpenMP static
    # schedule does: thread t owns a contiguous range of the N blocks.
    T: tl.constexpr = tl.dev.num_threads()
    NB: tl.constexpr = N // BLOCK
    scratch = tl.dev.alloc((T, BLOCK), tl.float32, placement="local")
    tl.store(threads_ptr, T)
    offs = tl.arange(0, BLOCK)
    for t in tl.dev.parallel(0, T):
        mine = scratch + t * BLOCK
        start = t * (NB // T) + tl.minimum(t, NB % T)
        count = NB // T + (t < NB % T).to(tl.int32)
        for b in range(start, start + count):
            tl.store(mine + offs, (b * BLOCK + offs).to(tl.float32))
            tl.store(out_ptr + b * BLOCK + offs, tl.load(mine + offs) * 2.0 + t)


@pytest.mark.parametrize("num_warps", [1, 3, 4])
def test_num_threads(device, num_warps):
    n, block = 20 * 16, 16
    out = torch.empty(n)
    threads = torch.zeros(1, dtype=torch.int32)
    compiled = _per_thread_kernel[(1,)](out, threads, N=n, BLOCK=block, num_warps=num_warps)
    assert threads.item() == num_warps == compiled.metadata.num_warps
    # Block b belongs to the thread whose contiguous range contains it.
    nb = n // block
    owner = [t for t in range(num_warps)
             for _ in range(nb // num_warps + (t < nb % num_warps))]
    ref = torch.cat([torch.arange(b * block, (b + 1) * block, dtype=torch.float32) * 2.0
                     + owner[b] for b in range(nb)])
    torch.testing.assert_close(out, ref)


def test_num_threads_default_is_one_cluster(device):
    from triton.backends.triton_shared import board

    out = torch.empty(64)
    threads = torch.zeros(1, dtype=torch.int32)
    compiled = _per_thread_kernel[(1,)](out, threads, N=64, BLOCK=16)
    assert threads.item() == compiled.metadata.num_warps == board.cluster_size(
        compiled.metadata.board_config)


# --------------------------------------------------------------------------
# tl.dev.warp_id
# --------------------------------------------------------------------------


@triton.jit
def _warp_kernel(warp_ptr, cpu_ptr, tid_ptr, N: tl.constexpr):
    pid = tl.program_id(0)
    for i in tl.dev.parallel(0, N):
        tl.store(warp_ptr + pid * N + i, tl.dev.warp_id())
        tl.store(cpu_ptr + pid * N + i, tl.dev.current_cpu())
        tl.store(tid_ptr + pid * N + i, tl.dev.os_thread_id())


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("num_warps", [1, 3, 4])
def test_warp_id(device, num_warps):
    from triton.backends.triton_shared import cpu_runtime

    grid, n = 3, 24
    warp = torch.full((grid, n), -1, dtype=torch.int32)
    cpu = torch.zeros((grid, n), dtype=torch.int32)
    tid = torch.zeros((grid, n), dtype=torch.int32)
    _warp_kernel[(grid,)](warp, cpu, tid, N=n, num_warps=num_warps)
    slots = [cpu_runtime.slot_cpus(num_warps, s) for s in range(cpu_runtime.num_slots(num_warps))]
    for p in range(grid):
        ws, cs, ts = warp[p].tolist(), cpu[p].tolist(), tid[p].tolist()
        slot = next(s for s in slots if cs[0] in s)
        assert set(ws) <= set(range(num_warps))
        for w, c, t in zip(ws, cs, ts):
            # The warp is the position of the CPU in the slot, one OS thread each.
            assert slot[w] == c
            assert t == ts[ws.index(w)]
