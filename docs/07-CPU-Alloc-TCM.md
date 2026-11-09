# Scratch buffers in the TCM (`cpu.alloc`)

`triton.language.extra.cpu.alloc(shape, dtype)` gives a kernel a scratch buffer
in the TCM (tightly coupled memory) of the thread executing the call.  On the
SpacemiT K3 every A100 core has a 384 KB TCM block next to it; elsewhere the
buffer is in ordinary memory of the thread, so the same kernel runs (with the
same results) on every platform.

## Usage

```python
from triton.language.extra import cpu

@triton.jit
def kernel(b_ptr, ..., K: tl.constexpr, BLOCK_N: tl.constexpr, ...):
    rk = tl.arange(0, K)[:, None]
    rn = tl.arange(0, BLOCK_N)[None, :]
    bt = cpu.alloc((K, BLOCK_N), tl.float32)        # pointer to K*BLOCK_N floats
    tl.store(bt + rk * BLOCK_N + rn, tl.load(...))  # the leader fills it
    for i in cpu.parallel(0, M, BLOCK_M):           # both threads read it
        b = tl.load(bt + rk * BLOCK_N + rn)
        tmp = cpu.alloc((BLOCK_M, BLOCK_N), tl.float32)  # one per thread
        ...

kernel[grid](..., num_warps=2)
```

* `shape` (compile-time integers) and `dtype` give the size: `prod(shape)`
  contiguous elements.  The result is a plain pointer (`tl.pointer_type(dtype)`)
  to the first element, 128-byte aligned; address it like any other pointer
  (wrap it with `tl.make_block_ptr` if a block pointer is wanted).
* **Owner.** The buffer is in the TCM of the thread that executes the call.
  Outside of a `cpu.parallel` loop that is the leader of the program instance,
  and the buffer is shared by the threads of the loop; inside the loop every
  thread gets its own buffer.  Data written by the leader before the loop is
  visible to all threads of the loop; concurrent writes to the same element
  are a data race, as for global memory.
* **Lifetime.** One program instance (or one iteration of the parallel loop).
  The contents are undefined on allocation, nothing has to be freed, and an
  allocation in a sequential loop returns the same address in every iteration.
* **Capacity.** All `cpu.alloc` buffers of a kernel (each rounded up to 128
  bytes) must fit into one TCM block, 384 KB; compilation fails otherwise and
  lists the buffers.  Buffers do not reuse each other's space.
* **Not supported:** atomic operations on the buffer (the TCM has none: an
  AMO or LR/SC on it raises SIGSEGV), sizes that are not compile-time constants,
  passing the pointer to anything else than loads, stores, pointer arithmetic
  and `.to(tl.int64)`.

### Shared buffers and `num_warps`

Only the two cores in a block's `cpu_affinity_mask` access it at full speed.
From any other core, even one in the same L2 cluster, reads are about 140x
slower (0.31 GB/s instead of 43 GB/s, see below).  A buffer allocated outside
of a `cpu.parallel` loop and used inside it is therefore only fast with
`num_warps=2`: the slots of the runtime are then `{8,9}`, `{10,11}`, `{12,13}`,
`{14,15}`, exactly the pairs sharing their blocks.  With other values the
runtime warns once (`... accesses slowly; use num_warps=2`); results stay
correct.

## Implementation

```
cpu.alloc                 tt.extern_elementwise "__triton_cpu_tcm_alloc"(bytes, 128) -> i64
                          tt.int_to_ptr
  triton-shared-opt       ... tptr.inttoptr / tptr.ptradd / tptr.to_memref
  --cpu-tcm-alloc-to-memref   memref.alloc() {alignment = 128, cpu.tcm} : memref<BYTESxi8>
                              + memref.view per access
  buddy-opt (vector)      bufferization / cpar passes leave it in place (inside or
                          outside of cpar.parallel)
  buddy-opt --cpu-tcm-assign  static offset per buffer (program order, aligned),
                              llvm.call @triton_cpu_tcm_base(total) + GEP where the
                              buffer was allocated, executed by that thread
  runtime                 triton_cpu_tcm_base: the TCM block of the calling worker
```

* `language/cpu/memory.py`: the builtin.  The call has side effects, so it is
  neither merged nor hoisted.
* `lib/Conversion/TritonToLinalgExperimental/CpuTcmAllocToMemrefPass.cpp`:
  runs at the end of `--triton-to-linalg-experimental`.  The buffer is marked
  with a plain attribute (`cpu.tcm`), not a memory space, so that views and the
  vector lowering are unaffected.
* `buddy-mlir/midend/lib/Conversion/CPar/TcmAssignPass.cpp`: added by
  `compiler.py` only for kernels containing `cpu.tcm`.  Buffers allocated
  outside of a `cpar.parallel` region and used inside it get their base from
  `triton_cpu_tcm_base_shared`, which checks the slot (see above).  The limit
  is 384 KB, `TRITON_CPU_TCM_LIMIT` overrides it.
* Runtime (`backend/runtime/triton_cpu_rt.cpp`, section "TCM"):
  * libspine_tcm (spacemit-tcm 3.x) is loaded with `dlopen`; the runtime does
    not depend on it.
  * A worker acquires its block on its first `triton_cpu_tcm_base` call, with
    `spine_tcm_mem_try_wait(block, 0)` (exclusive, non-blocking; `mem_get` is
    only a reference count and does not exclude other processes).  The block
    is the one whose mask contains the worker's core; of the two blocks of a
    pair the lower core gets the lower one (cpu8 → 0, …, cpu15 → 7).
  * Blocks are released at exit by the workers themselves (the library only
    accepts a release from the acquiring thread), or by `tcrt_tcm_release()`.
  * A block left behind by a dead thread (process killed while holding it) is
    reclaimed with `mem_force_release`, with a warning.  A block held by a live
    thread of another process is not touched: the worker uses ordinary memory
    and the runtime warns once.
  * Ordinary memory is used as well without AI cores (the X100 cores do not
    write the TCM reliably: `memset` leaves most of it unchanged), without
    libspine_tcm, when the buffers exceed a block, and with `TRITON_CPU_TCM=0`.
    It is a 128-byte aligned per-thread buffer of at least 384 KB.
  * After `fork()` the child's new workers cannot take the blocks still held by
    the parent's workers and use ordinary memory.
  * Nested launches (from a worker) would reuse the TCM of the enclosing
    instance; they do not occur with the generated launchers.

| Environment variable | Meaning |
|---|---|
| `TRITON_CPU_TCM` | unset: use the TCM when available, else ordinary memory. `0`: always ordinary memory. `require`: a launch in which some buffer was not in the TCM raises `RuntimeError` (after executing it). Read when a worker first uses `cpu.alloc` and at every launch. |
| `TRITON_CPU_TCM_LIMIT` | Capacity in bytes checked at compile time (default 393216). |

Introspection (`triton_cpu_rt.h`): `tcrt_tcm_available()`,
`tcrt_tcm_block_bytes()`, `tcrt_worker_tcm(cpu, &block, &base)`,
`tcrt_tcm_release()`.

## Why not `smt.alloc` (spine-triton)

spine-triton maps `scope="tcm"` to the memory space
`#xsmt.memory_space<thread_local>` (documented as compiler-managed scratch)
and lowers it to a plain `memref.alloc`; placing it is left to the closed
spine-opt, run with `enable-always-tls=1`, and nothing in spine-triton
acquires a TCM block (no libspine_tcm call).  Its `alloc_copies` views also
drop the scope (`BufferTensorViewOp::build` hard-codes address space 1).  Here
the allocation is resolved to a runtime call by our own passes, and the runtime
owns the blocks.

## Measurements (SpacemiT K3, A100 at 1.8 GHz)

C programs `tmp/tcm_{sem,access,bw}.c`, RVV e32m8, buffers of 384 KB:

| Memory | Core | Read GB/s | Write GB/s | Random latency |
|---|---|---|---|---|
| DRAM, L2 resident | cpu8 | 45.9 | 50.8 | 19.4 ns |
| DRAM, 64 MB | cpu8 | 7.1 | 26.3 | 183 ns |
| TCM, core in the block's mask | cpu8, cpu9 → block 0 | 43-44 | 51.9 | 9.4 ns |
| TCM, core not in the mask | cpu10, cpu12 → block 0 | 0.31 | 3.7 | 57 ns |

Four threads on `cpu8-11` reading at the same time: each its own nearby block
170 GB/s in total, a shared L2 resident DRAM buffer 93 GB/s (L2 limit).  The
TCM pays off through per-core bandwidth, not through sharing.

With the code currently generated by the pipeline the kernels are far from
these limits, and the placement of the buffer hardly shows
(`tmp/bench_tcm_{stream,matmul}.py`, grid 8 / 16):

| Kernel | Global input | `cpu.alloc` in TCM | `cpu.alloc`, `TRITON_CPU_TCM=0` |
|---|---|---|---|
| Re-read 256 KB per instance 20 times (W=1, 8 instances) | 53.8 ms (0.78 GB/s) | 44.9 ms (0.93 GB/s) | 46.6 ms (0.90 GB/s) |
| Matmul 1024x256x512, B tile packed (W=2) | 1001 ms | 994 ms | 990 ms |

Every `tl.load` becomes a `malloc` + `memref.copy` into a temporary (never
freed) followed by the computation, so the kernels run at about 1 GB/s and
0.27 GFLOP/s.  The TCM can only make a difference once loads are vectorized in
place (and with A100 matrix instructions).

## Verification

`python/examples/test_cpu_alloc.py` (25 tests; the 6 TCM specific ones are
skipped without TCM):

* read/write (1-d, 2-d transpose, scalar, int8 … int64, float32, gather);
  layout of several buffers (disjoint, aligned); private buffers in parallel
  loops (one address per thread, values not mixed up); shared buffer outside
  of the loop; same buffer in every iteration of a sequential loop; capacity
  error; non-constant shape rejected;
* K3: every worker's buffers are inside the block next to its core, distinct
  blocks per core; shared buffers in the leader's block; `TRITON_CPU_TCM=0`;
  another process holding all blocks (fallback with warning, `require` raises);
  the `num_warps=4` warning (and none with 2); `fork()`.

lit: `test/Conversion/CpuTcmAllocToMemref/*.mlir`,
`test/Conversion/TritonToUnstructured/make_tptr_static_offsets.mlir`,
`buddy-mlir/tests/Conversion/CPar/cpu-tcm-assign.mlir`.
