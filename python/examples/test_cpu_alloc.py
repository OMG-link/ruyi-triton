"""triton.language.extra.cpu.alloc: scratch buffers in the TCM of the thread.

``cpu.alloc(shape, dtype)`` returns a pointer to ``prod(shape)`` elements in
the TCM of the thread executing the call (SpacemiT K3, A100 cores), or in
ordinary memory of that thread elsewhere.  The tests check

* reads and writes through the pointer (1-d, 2-d, scalar, several dtypes),
* the layout of several buffers (disjoint, 128-byte aligned),
* buffers allocated inside a ``cpu.parallel`` loop are private to each thread,
  buffers allocated outside of it are shared by the threads of the instance,
* a buffer allocated in a sequential loop is the same in every iteration,
* the compile-time capacity check,
* on the K3: every worker uses the TCM block next to its core, the buffers are
  in that block, TRITON_CPU_TCM=0 / =require, another process holding the
  blocks, the warning for shared buffers with num_warps=4, fork().
"""

import ctypes
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest
import torch

import triton
import triton.language as tl
from triton.language.extra import cpu

from triton.backends.triton_shared import driver as cpu_driver
from triton.backends.triton_shared.driver import CPUDriver

TCM_ENV = "TRITON_CPU_TCM"
ALIGN = 128


@pytest.fixture
def backend(monkeypatch):
    triton.runtime.driver.set_active(CPUDriver())
    monkeypatch.delenv(TCM_ENV, raising=False)
    monkeypatch.delenv("TRITON_CPU_NUM_THREADS", raising=False)
    return "cpu"


def runtime():
    path, _ = cpu_driver._get_runtime_library(cpu_driver._get_sanitizer_type())
    lib = ctypes.CDLL(path)
    lib.tcrt_tcm_block_bytes.restype = ctypes.c_int64
    lib.tcrt_worker_tcm.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_void_p)]
    return lib


def tcm_available():
    try:
        return runtime().tcrt_tcm_available() == 1
    except OSError:
        return False


needs_tcm = pytest.mark.skipif(not tcm_available(), reason="needs the TCM of the SpacemiT K3 (AI cores, libspine_tcm)")


def worker_tcm(cpu_id):
    """(rc, block, base) of the worker on `cpu_id`, see tcrt_worker_tcm."""
    block = ctypes.c_int(-9)
    base = ctypes.c_void_p()
    rc = runtime().tcrt_worker_tcm(cpu_id, ctypes.byref(block), ctypes.byref(base))
    return rc, block.value, base.value or 0


class _BlockInfo(ctypes.Structure):
    # spine_tcm_block_info_t of spine_tcm.h
    _fields_ = [("id", ctypes.c_int), ("va", ctypes.c_void_p), ("size", ctypes.c_size_t),
                ("phys_addr", ctypes.c_uint64), ("cpu_affinity_mask", ctypes.c_uint64),
                ("owner_tid", ctypes.c_int), ("is_acquired", ctypes.c_int)]


def block_mask(block):
    lib = ctypes.CDLL("libspine_tcm.so")
    info = _BlockInfo()
    assert lib.spine_tcm_runtime_mem_info(block, ctypes.byref(info)) == 0
    return info.cpu_affinity_mask


# ------------------------------------------------------------------ kernels


@triton.jit
def roundtrip_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    buf = cpu.alloc((BLOCK, ), tl.float32)
    offs = tl.arange(0, BLOCK)
    tl.store(buf + offs, tl.load(x_ptr + offs) * 2.0)
    tl.store(out_ptr + offs, tl.load(buf + offs) + 1.0)


@triton.jit
def transpose_kernel(x_ptr, out_ptr, M: tl.constexpr, N: tl.constexpr):
    buf = cpu.alloc((M, N), tl.float32)
    rm = tl.arange(0, M)[:, None]
    rn = tl.arange(0, N)[None, :]
    tl.store(buf + rm * N + rn, tl.load(x_ptr + rm * N + rn))
    tl.store(out_ptr + rn * M + rm, tl.load(buf + rm * N + rn))


@triton.jit
def dtype_kernel(x_ptr, out_ptr, BLOCK: tl.constexpr):
    buf = cpu.alloc((BLOCK, ), x_ptr.dtype.element_ty)
    offs = tl.arange(0, BLOCK)
    tl.store(buf + offs, tl.load(x_ptr + offs))
    tl.store(out_ptr + offs, tl.load(buf + BLOCK - 1 - offs))


@triton.jit
def scalar_kernel(x_ptr, out_ptr):
    buf = cpu.alloc((4, ), tl.float32)
    tl.store(buf + 3, tl.load(x_ptr) + 7.0)
    tl.store(buf, tl.load(x_ptr) - 1.0)
    tl.store(out_ptr, tl.load(buf + 3))
    tl.store(out_ptr + 1, tl.load(buf))


@triton.jit
def layout_kernel(addr_ptr):
    a = cpu.alloc((3, ), tl.int8)
    b = cpu.alloc((5, 7), tl.float32)
    c = cpu.alloc((1, ), tl.int64)
    tl.store(a, 1)
    tl.store(b, 2.0)
    tl.store(c, 3)
    tl.store(addr_ptr + 0, a.to(tl.int64))
    tl.store(addr_ptr + 1, b.to(tl.int64))
    tl.store(addr_ptr + 2, c.to(tl.int64))
    tl.store(addr_ptr + 3, tl.load(a).to(tl.int64) + tl.load(b).to(tl.int64) + tl.load(c))


@triton.jit
def private_kernel(out_ptr, addr_ptr, cpu_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr, SPIN: tl.constexpr):
    """Every iteration writes its own values into a buffer allocated inside the
    parallel loop, works a while and reads them back."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    for i in cpu.parallel(0, ITERS):
        buf = cpu.alloc((BLOCK, ), tl.float32)
        k = pid * ITERS + i
        tl.store(buf + offs, (offs + k * BLOCK).to(tl.float32))
        w = offs.to(tl.float32)
        for _ in range(SPIN):
            w = w * 1.0000001 + 1.0
        tl.store(out_ptr + k * BLOCK + offs, tl.load(buf + offs) + w * 0.0)
        tl.store(addr_ptr + k, buf.to(tl.int64))
        tl.store(cpu_ptr + k, cpu.current_cpu())


@triton.jit
def shared_kernel(x_ptr, out_ptr, addr_ptr, ITERS: tl.constexpr, BLOCK: tl.constexpr):
    """A table allocated and filled by the leader, read by all threads."""
    pid = tl.program_id(0)
    offs = tl.arange(0, BLOCK)
    table = cpu.alloc((BLOCK, ), tl.float32)
    tl.store(table + offs, tl.load(x_ptr + pid * BLOCK + offs))
    tl.store(addr_ptr + pid, table.to(tl.int64))
    for i in cpu.parallel(0, ITERS):
        tl.store(out_ptr + (pid * ITERS + i) * BLOCK + offs, tl.load(table + offs) + i)


@triton.jit
def serial_kernel(out_ptr, addr_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    for i in range(N):
        buf = cpu.alloc((BLOCK, ), tl.int32)
        tl.store(buf + offs, offs + i)
        tl.store(out_ptr + i * BLOCK + offs, tl.load(buf + offs))
        tl.store(addr_ptr + i, buf.to(tl.int64))


@triton.jit
def too_big_kernel(out_ptr):
    a = cpu.alloc((300 * 1024, ), tl.int8)
    b = cpu.alloc((100 * 1024, ), tl.int8)
    tl.store(a, 1)
    tl.store(b, 2)
    tl.store(out_ptr, tl.load(a) + tl.load(b))


@triton.jit
def dynamic_shape_kernel(out_ptr, n):
    buf = cpu.alloc((n, ), tl.float32)
    tl.store(buf, 1.0)
    tl.store(out_ptr, tl.load(buf))


def run_private(grid, warps, iters=8, block=32, spin=2000):
    n = grid * iters
    out = torch.full((n * block, ), -1.0)
    addrs = torch.zeros(n, dtype=torch.int64)
    cpus = torch.full((n, ), -1, dtype=torch.int32)
    private_kernel[(grid, )](out, addrs, cpus, ITERS=iters, BLOCK=block, SPIN=spin, num_warps=warps)
    torch.testing.assert_close(out, torch.arange(n * block, dtype=torch.float32))
    return addrs.tolist(), cpus.tolist()


def run_shared(grid, warps, iters=8, block=32):
    x = torch.randn(grid, block)
    out = torch.empty(grid, iters, block)
    addrs = torch.zeros(grid, dtype=torch.int64)
    shared_kernel[(grid, )](x, out, addrs, ITERS=iters, BLOCK=block, num_warps=warps)
    expected = x[:, None, :] + torch.arange(iters, dtype=torch.float32)[None, :, None]
    torch.testing.assert_close(out, expected)
    return addrs.tolist()


# ---------------------------------------------------- tests (all platforms)


def test_roundtrip(backend):
    x = torch.randn(64)
    out = torch.empty(64)
    roundtrip_kernel[(1, )](x, out, BLOCK=64)
    torch.testing.assert_close(out, x * 2 + 1)


def test_2d(backend):
    x = torch.randn(16, 32)
    out = torch.empty(32, 16)
    transpose_kernel[(1, )](x, out, M=16, N=32)
    torch.testing.assert_close(out, x.T)


@pytest.mark.parametrize("dtype", [torch.int8, torch.int16, torch.int32, torch.int64, torch.float32])
def test_dtypes(backend, dtype):
    x = (torch.randn(64) * 50).to(dtype)
    out = torch.empty_like(x)
    dtype_kernel[(1, )](x, out, BLOCK=64)
    torch.testing.assert_close(out, x.flip(0))


def test_scalar(backend):
    x = torch.randn(1)
    out = torch.empty(2)
    scalar_kernel[(1, )](x, out)
    torch.testing.assert_close(out, torch.cat([x + 7, x - 1]))


def test_layout(backend):
    """Buffers follow each other in program order, each 128-byte aligned."""
    addrs = torch.zeros(4, dtype=torch.int64)
    layout_kernel[(1, )](addrs)
    a, b, c, total = addrs.tolist()
    assert total == 6
    assert a % ALIGN == 0 and b % ALIGN == 0 and c % ALIGN == 0, [hex(a), hex(b), hex(c)]
    assert b - a == ALIGN  # 3 bytes, rounded up
    assert c - b == 2 * ALIGN  # 140 bytes, rounded up


@pytest.mark.parametrize("grid,warps", [(1, 2), (8, 2), (4, 4), (64, 1)])
def test_private_buffers_in_parallel_loop(backend, grid, warps):
    addrs, cpus = run_private(grid, warps)
    # One buffer per thread: the same CPU always gets the same address, two
    # CPUs never share one.
    by_cpu = {}
    for addr, c in zip(addrs, cpus):
        by_cpu.setdefault(c, set()).add(addr)
    assert all(len(a) == 1 for a in by_cpu.values()), by_cpu
    owners = [next(iter(a)) for a in by_cpu.values()]
    assert len(set(owners)) == len(owners), by_cpu


@pytest.mark.parametrize("grid,warps", [(1, 2), (8, 2), (16, 1)])
def test_shared_buffer_outside_parallel_loop(backend, grid, warps):
    run_shared(grid, warps)


def test_sequential_loop_reuses_buffer(backend):
    n, block = 6, 16
    out = torch.empty(n * block, dtype=torch.int32)
    addrs = torch.zeros(n, dtype=torch.int64)
    serial_kernel[(1, )](out, addrs, N=n, BLOCK=block)
    expected = (torch.arange(block)[None, :] + torch.arange(n)[:, None]).flatten().to(torch.int32)
    torch.testing.assert_close(out, expected)
    assert len(set(addrs.tolist())) == 1


def test_capacity_exceeded(backend, capfd):
    out = torch.empty(1, dtype=torch.int8)
    with pytest.raises(Exception):
        too_big_kernel[(1, )](out)
    err = capfd.readouterr().err
    assert "cpu.alloc: the buffers of this kernel need 409600 bytes of TCM" in err, err


def test_shape_must_be_constant(backend):
    out = torch.empty(1)
    with pytest.raises(Exception, match="compile-time"):
        dynamic_shape_kernel[(1, )](out, 4)


# ------------------------------------------------------------ tests (K3 TCM)


@needs_tcm
def test_workers_use_their_tcm_block(backend):
    """Every buffer is in the TCM block next to the core of its thread."""
    block_bytes = runtime().tcrt_tcm_block_bytes()
    assert block_bytes > 0
    addrs, cpus = run_private(8, 2)
    blocks = {}
    for addr, c in zip(addrs, cpus):
        rc, block, base = worker_tcm(c)
        assert rc == 1, f"worker on cpu {c} has no TCM block"
        assert base <= addr < base + block_bytes, (c, hex(addr), hex(base))
        assert (block_mask(block) >> c) & 1, f"cpu {c} uses block {block}"
        blocks[c] = block
    assert len(set(blocks.values())) == len(blocks), blocks
    print(f"cpu -> TCM block: {dict(sorted(blocks.items()))}")


@needs_tcm
def test_shared_buffer_in_leader_tcm(backend):
    block_bytes = runtime().tcrt_tcm_block_bytes()
    addrs = run_shared(8, 2)
    bases = []
    for c in range(64):
        rc, _, base = worker_tcm(c)
        if rc == 1:
            bases.append(base)
    for addr in addrs:
        assert any(base <= addr < base + block_bytes for base in bases), hex(addr)


def run_script(tmp_path, body, env=None, timeout=600):
    # The blocks taken by the workers of this process are not available to the
    # script; they are acquired again by the next kernel here.
    runtime().tcrt_tcm_release()
    script = tmp_path / "script.py"
    script.write_text(
        textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).parent)!r})
        import triton
        from triton.backends.triton_shared.driver import CPUDriver
        triton.runtime.driver.set_active(CPUDriver())
        import test_cpu_alloc as t
    """) + textwrap.dedent(body))
    full_env = dict(os.environ)
    full_env.pop(TCM_ENV, None)
    full_env.update(env or {})
    # cwd: not next to a triton source tree, which would shadow the package.
    return subprocess.run([sys.executable, str(script)], env=full_env, cwd=tmp_path, capture_output=True, text=True,
                          timeout=timeout)


@needs_tcm
def test_tcm_disabled(backend, tmp_path):
    out = run_script(tmp_path, """
        addrs, cpus = t.run_private(8, 2)
        print("rc", sorted({t.worker_tcm(c)[0] for c in cpus}))
    """, env={TCM_ENV: "0"})
    assert out.returncode == 0, out.stderr
    assert "rc [0]" in out.stdout, out.stdout


@needs_tcm
def test_blocks_held_by_another_process(backend, tmp_path):
    """Correct results in ordinary memory (with a warning); an error with
    TRITON_CPU_TCM=require."""
    runtime().tcrt_tcm_release()
    holder = subprocess.Popen(
        [sys.executable, "-c", textwrap.dedent("""
            import ctypes, sys
            lib = ctypes.CDLL("libspine_tcm.so")
            lib.spine_tcm_runtime_mem_try_wait.restype = ctypes.c_void_p
            held = [i for i in range(8) if lib.spine_tcm_runtime_mem_try_wait(i, 0)]
            print(len(held), flush=True)
            sys.stdin.readline()
            for i in held:
                lib.spine_tcm_runtime_mem_release(i)
        """)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, cwd=tmp_path)
    try:
        assert holder.stdout.readline().strip() == "8", "could not take all TCM blocks"
        out = run_script(tmp_path, """
            addrs, cpus = t.run_private(8, 2)
            print("rc", sorted({t.worker_tcm(c)[0] for c in cpus}))
        """)
        assert out.returncode == 0, out.stderr
        assert "rc [0]" in out.stdout, out.stdout
        assert "is used by thread" in out.stderr, out.stderr

        out = run_script(tmp_path, """
            try:
                t.run_private(8, 2)
            except RuntimeError as e:
                print("error", e)
        """, env={TCM_ENV: "require"})
        assert out.returncode == 0, out.stderr
        assert "error cpu.alloc buffers could not be put in the TCM" in out.stdout, out.stdout
    finally:
        holder.stdin.write("\n")
        holder.stdin.flush()
        holder.wait(timeout=60)


@needs_tcm
def test_shared_buffer_warning(backend, tmp_path):
    """num_warps=4 puts threads away from the leader's TCM block: warned."""
    out = run_script(tmp_path, "t.run_shared(4, 4)\n")
    assert out.returncode == 0, out.stderr
    assert "accesses slowly; use num_warps=2" in out.stderr, out.stderr
    out = run_script(tmp_path, "t.run_shared(4, 2)\n")
    assert out.returncode == 0, out.stderr
    assert "accesses slowly" not in out.stderr, out.stderr


@needs_tcm
@pytest.mark.filterwarnings("ignore:This process .* is multi-threaded:DeprecationWarning")
def test_fork(backend):
    """A forked child (whose new workers cannot take the parent's blocks)
    still computes correct results."""
    run_private(8, 2)
    pid = os.fork()
    if pid == 0:  # child
        code = 1
        try:
            run_private(8, 2)
            code = 0
        finally:
            os._exit(code)
    _, status = os.waitpid(pid, 0)
    assert os.WIFEXITED(status) and os.WEXITSTATUS(status) == 0, status
