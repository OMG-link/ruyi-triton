# Threads per program instance

A Triton program instance corresponds to a GPU thread block, which contains
several warps. On the CPU backend a program instance can run a loop with
several threads that stay on one cluster and share its L2. It builds on
[program instance parallelism](09-Program-Instance-Parallelism.md).

## Usage

```python
@triton.jit
def kernel(x_ptr, out_ptr, sum_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    for i in tl.dev.parallel(0, N, BLOCK):      # num_warps threads
        offs = base + i + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) * 2.0)
    tl.store(sum_ptr + pid,                      # one thread, after the join
             tl.sum(tl.load(out_ptr + base + tl.arange(0, N))))

kernel[(grid,)](x, out, sums, N=4096, BLOCK=128, num_warps=4)
```

* `tl.dev.parallel(start, end, step)` is used like `tl.range`. Its
  iterations are independent; thread `t` of the instance runs iterations
  `t, t + W, t + 2W, ...`, and the code after the loop starts when all
  threads are done (fork/join). Code outside such loops runs on one thread.
* Values defined before the loop (pointers, scalars) are shared by the
  threads. Tensors created in the body belong to the iteration.
* `tl.dev.alloc` sits directly in the kernel body, so its buffers are
  allocated before the loop and shared by the threads of the instance,
  whatever the placement; an `spm` buffer lives in the instance's slot share,
  like a GPU's shared memory. Per-thread scratch is a buffer `num_warps`
  times as large, indexed by the thread that runs the iteration. Thread `t`
  runs iterations `t, t + W, ...`, so that thread is `(i // step) % W` for
  the iteration starting at `start + i`:

  ```python
  scratch = tl.dev.alloc((W, BLOCK), tl.float32, placement="local")
  for i in tl.dev.parallel(0, N, BLOCK):
      mine = scratch + ((i // BLOCK) % W) * BLOCK   # this thread's row
      ...
  ```

  `W` is the kernel's resolved `num_warps`; pass it as a `tl.constexpr`
  equal to `num_warps`.
* The body must not assign to a variable defined before the loop: there is
  no value to carry between iterations (compile error). Writes of different
  iterations to the same element race; nothing checks for it.
* Nested `tl.dev.parallel` loops: only the outermost one is distributed,
  inner ones run sequentially in each thread.

## `num_warps`

`num_warps` is the number of threads W of a program instance: used as is
when > 0, and the size of the first domain of the kernel's board
configuration when 0 (the default), i.e. one cluster per instance. The
resolved value is the kernel's `num_warps` and part of its cache key; a
different `num_warps` recompiles.

The runtime gives a launch W CPUs per instance only if the kernel forks a
parallel region (`metadata.cpu_threads` is W, otherwise 1). A kernel without
`tl.dev.parallel` keeps one CPU per instance whatever `num_warps` says, so
`num_warps` passed out of GPU habit costs no parallelism.

## Pipeline

```
tl.dev.parallel loop
  │ Triton code generator (patch 0015): scf.forall (%iv) = (lb) to (ub) step (s)
  ▼
TTIR ─ triton-shared-opt ─► linalg on tensors, the scf.forall unchanged
  │ buddy-opt --cpar-distribute-loops=num-threads=W   (before bufferization)
  ▼
scf.forall (%t) in (W) {
  scf.for %iv = lb + t*s to ub step s*W { body }
} {cpar.threads}
  │ --one-shot-bufferize: the thread forall is a parallel region, tensors
  │ written inside get private buffers; --buffer-deallocation-pipeline
  │ --cpar-forall-to-parallel
  ▼
cpar.parallel num_threads(W) { ^bb0(%tid, %nt): scf.for ... cpar.yield }
  │ lowering to the LLVM dialect, the region stays nested
  │ --cpar-outline-to-runtime                          (last)
  ▼
llvm.func internal @kernel__cpar0(%ctx, %tid, %nt) { load captures; body }
llvm.call @__cpar_parallel(@kernel__cpar0, %ctx, W)
```

Both the IME and the standard pipeline of `backend/compiler.py` run these
passes. `--cpar-distribute-loops` with W = 1 turns the loop into a plain
`scf.for`, so no region is created.

The detour through a thread forall matters: bufferizing a sequential loop
may legally reuse one buffer for a temporary of all iterations (e.g. when
CSE hoisted the `tensor.empty` out of the loop). Run by several threads,
that buffer would be shared.

`cpar.parallel` is an automatic allocation scope, so stack buffers inside it
are per thread. buddy-mlir also exposes `wrapInParallelRegion`,
`distributeLoop` and `distributeForall` (`CPar/CParTransforms.h`) for passes
that create parallel regions themselves.

## Runtime ABI

```c
void __cpar_parallel(void (*fn)(void *ctx, int64_t tid, int64_t nthreads),
                     void *ctx, int64_t nthreads);
```

It calls `fn(ctx, tid, nthreads)` for every `tid` in `[0, nthreads)` and
returns when all calls are done. Called by the leader of a slot of S CPUs,
it runs the ids `0, S, 2S, ...` itself and posts the region to the other
workers of the slot (helper `m` runs `m, m + S, ...`). Called anywhere else
(single CPU slot, nested region, outside a launch) it runs the ids one after
the other on the calling thread. The kernel always sees `nthreads = W`.

## Trace

`<TRITON_CPAR_TRACE_FILE>.threads`: one row per call `fn(ctx, tid, nthreads)`,
`instance,region,tid,nthreads,os_tid,cpu,start_ns,end_ns`.

## Constraints

* No synchronization inside a region and no reduction across threads:
  reduce per thread into memory and combine after the loop.
* W is a compile-time value.
* Captured values are passed through a context struct; `llvm.noalias` of
  the kernel arguments does not reach the outlined function (see TODO).

## Tests

`python/examples/test_program_instance_parallel.py`; buddy-mlir lit tests
`tests/Dialect/CPar/` and `tests/Conversion/CPar/`; triton-shared lit test
`test/Conversion/TritonToLinalgExperimental/parallel_forall.mlir`.
