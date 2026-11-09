"""Kernels run on the AI cores of the SpacemiT K3.

On the K3 the A100 ("AI") cores are outside the affinity of ordinary threads.
A thread gets access to them, and loses access to the other cores, by writing
its id to /proc/set_ai_thread.  The CPU runtime (backend/runtime) uses this by
default: its worker pool runs on the AI cores only, and no kernel code runs
anywhere else, not even on the thread that launches the kernel.
TRITON_CPU_AI_CORES=0 restores the pool on the CPUs of the process.

The tests check that

* the pool's CPUs are the AI cores and disjoint from the affinity of the
  launching thread,
* program instances and parallel regions run only on AI cores, on threads
  pinned to one AI core each, also for grid=1 / num_warps=1,
* the threads of an instance stay in one L2 domain of AI cores,
* the launching thread is not moved,
* the pool is rebuilt on the AI cores after fork(),
* TRITON_CPU_AI_CORES=0 runs the kernels on the CPUs of the process.

Skipped where /proc/set_ai_thread does not exist (K1, x86, qemu).
"""

import ctypes
import os
import subprocess
import sys
import textwrap
import threading
from pathlib import Path

import pytest
import torch

import triton
import triton.language as tl
from triton.language.extra import cpu

from triton.backends.triton_shared import compiler as cpu_compiler
from triton.backends.triton_shared import driver as cpu_driver
from triton.backends.triton_shared.driver import CPUDriver

AI_THREAD_PATH = "/proc/set_ai_thread"
AI_CORES_ENV = "TRITON_CPU_AI_CORES"
NUM_THREADS_ENV = "TRITON_CPU_NUM_THREADS"
TRACE_FILE_ENV = "TRITON_CPU_TRACE_FILE"

pytestmark = pytest.mark.skipif(
    not os.access(AI_THREAD_PATH, os.W_OK) or os.getenv(AI_CORES_ENV) == "0",
    reason="needs the SpacemiT AI cores (/proc/set_ai_thread) with the AI mode enabled",
)


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
    lib.tcrt_domain_cpus.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.c_int]
    return lib


def runtime_domains():
    lib = runtime()
    domains = []
    for d in range(lib.tcrt_num_domains()):
        buf = (ctypes.c_int * 1024)()
        n = lib.tcrt_domain_cpus(d, buf, 1024)
        domains.append(frozenset(buf[:n]))
    return domains


def runtime_cpus():
    return frozenset().union(*runtime_domains())


def read_csv(path):
    lines = Path(path).read_text().splitlines()
    header = lines[0].split(",")
    return [dict(zip(header, map(int, line.split(",")))) for line in lines[1:] if line]


# ------------------------------------------------------------------ kernels


@triton.jit
def where_kernel(serial_ptr, region_ptr, tid_ptr, work_ptr, ITERS: tl.constexpr, SPIN: tl.constexpr):
    """Records the CPU of the serial part and of every parallel iteration."""
    pid = tl.program_id(0)
    tl.store(serial_ptr + pid, cpu.current_cpu())
    for i in cpu.parallel(0, ITERS):
        offs = tl.arange(0, 16)
        acc = (offs + i).to(tl.float32)
        for _ in range(SPIN):
            acc = acc * 1.0000001 + 1.0
        tl.store(work_ptr + (pid * ITERS + i) * 16 + offs, acc)
        tl.store(region_ptr + pid * ITERS + i, cpu.current_cpu())
        tl.store(tid_ptr + pid * ITERS + i, cpu.os_thread_id())


def run_where(grid, warps, iters=8, spin=2000):
    serial = torch.full((grid,), -1, dtype=torch.int32)
    region = torch.full((grid * iters,), -1, dtype=torch.int32)
    tids = torch.full((grid * iters,), -1, dtype=torch.int32)
    work = torch.zeros((grid * iters * 16,), dtype=torch.float32)
    where_kernel[(grid,)](serial, region, tids, work, ITERS=iters, SPIN=spin, num_warps=warps)
    return serial.tolist(), region.view(grid, iters).tolist(), tids.view(grid, iters).tolist()


# -------------------------------------------------------------------- tests


def test_pool_on_ai_cores(backend):
    """The pool's CPUs are the AI cores, outside the affinity of this thread."""
    lib = runtime()
    assert lib.tcrt_ai_cores() == 1
    cpus = runtime_cpus()
    caller = frozenset(os.sched_getaffinity(0))
    assert cpus and not (cpus & caller), f"pool {sorted(cpus)}, caller {sorted(caller)}"
    # The compiler sees the same CPUs (num_warps=0 inference).
    assert cpu_compiler._ai_core_cpus() == cpus
    for domain in runtime_domains():
        print(f"AI domain {sorted(domain)}")


@pytest.mark.parametrize("grid,warps", [(1, 1), (1, 4), (64, 1), (16, 4)])
def test_kernels_run_only_on_ai_cores(backend, monkeypatch, tmp_path, grid, warps):
    """Serial parts, parallel iterations and every thread involved are on AI cores."""
    cpus = runtime_cpus()
    trace = tmp_path / "trace.csv"
    monkeypatch.setenv(TRACE_FILE_ENV, str(trace))
    serial, region, tids = run_where(grid, warps)

    used = set(serial) | {c for row in region for c in row}
    assert used <= cpus, f"kernel code ran on {sorted(used - cpus)}"

    instances = read_csv(trace)
    regions = read_csv(str(trace) + ".threads")
    main_tid = threading.get_native_id()
    threads = {r["os_tid"] for r in instances} | {r["os_tid"] for r in regions}
    threads |= {t for row in tids for t in row}
    assert main_tid not in threads, "kernel code ran on the launching thread"
    for row in instances + regions:
        assert row["cpu"] in cpus, row
    for tid in threads:
        allowed = os.sched_getaffinity(tid)
        assert len(allowed) == 1 and allowed <= cpus, f"thread {tid} may run on {allowed}"

    if warps > 1:
        domains = runtime_domains()
        for pid in range(grid):
            on = set(region[pid]) | {serial[pid]}
            assert sum(on <= d for d in domains) == 1, f"instance {pid} on {sorted(on)}"
            assert len(set(tids[pid])) == min(warps, max(len(d) for d in domains))
    print(f"grid={grid} num_warps={warps}: CPUs {sorted(used)}")


def test_launching_thread_not_moved(backend):
    before = os.sched_getaffinity(0)
    run_where(4, 4)
    run_where(1, 1)
    assert os.sched_getaffinity(0) == before


@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_pool_rebuilt_after_fork(backend):
    """A forked child launches on a fresh pool, again on the AI cores."""
    cpus = runtime_cpus()
    run_where(8, 4)  # compile and start the pool in the parent
    read_fd, write_fd = os.pipe()
    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            os.close(read_fd)
            serial, region, _ = run_where(8, 4)
            used = set(serial) | {c for row in region for c in row}
            os.write(write_fd, ",".join(map(str, sorted(used))).encode())
            code = 0
        finally:
            os._exit(code)
    os.close(write_fd)
    with os.fdopen(read_fd) as pipe:
        reported = pipe.read()
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
    used = {int(c) for c in reported.split(",") if c}
    assert used and used <= cpus, f"child ran on {sorted(used)}"


def test_opt_out(backend, tmp_path):
    """TRITON_CPU_AI_CORES=0: the pool and the kernels use the process' CPUs."""
    script = tmp_path / "opt_out.py"
    script.write_text(textwrap.dedent(f"""
        import ctypes, os, sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        import triton
        from triton.backends.triton_shared import driver as cpu_driver
        triton.runtime.driver.set_active(cpu_driver.CPUDriver())
        from test_program_instance_ai_cores import run_where
        lib = ctypes.CDLL(cpu_driver._get_runtime_library(cpu_driver._get_sanitizer_type())[0])
        serial, region, _ = run_where(8, 4)
        used = set(serial) | {{c for row in region for c in row}}
        print("ai", lib.tcrt_ai_cores())
        print("used", ",".join(map(str, sorted(used))))
    """))
    env = dict(os.environ, **{AI_CORES_ENV: "0"})
    # cwd: not next to a triton source tree, which would shadow the package.
    out = subprocess.run([sys.executable, str(script)], env=env, cwd=tmp_path,
                         capture_output=True, text=True, timeout=600)
    assert out.returncode == 0, out.stderr
    lines = dict(line.split(" ", 1) for line in out.stdout.splitlines() if " " in line)
    assert lines["ai"] == "0"
    used = {int(c) for c in lines["used"].split(",")}
    assert used <= set(os.sched_getaffinity(0)), f"opt-out ran on {sorted(used)}"
    assert not (used & runtime_cpus())
    print(f"TRITON_CPU_AI_CORES=0: CPUs {sorted(used)}")
