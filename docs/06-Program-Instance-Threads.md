# Several threads per program instance

A Triton program instance corresponds to a GPU thread block, which contains
several warps.  On the CPU backend a program instance used to be a single
thread.  This document describes how a program instance now runs a code region
with several threads that stay on one cluster (and share its L2), and how the
number of threads is kept under control.  It builds on
[program instance level parallelism](05-Program-Instance-Parallelism.md).

## Usage

```python
from triton.language.extra import cpu

@triton.jit
def kernel(x_ptr, out_ptr, sum_ptr, N: tl.constexpr, BLOCK: tl.constexpr):
    pid = tl.program_id(0)
    base = pid * N
    for i in cpu.parallel(0, N, BLOCK):        # num_warps threads
        offs = base + i + tl.arange(0, BLOCK)
        tl.store(out_ptr + offs, tl.load(x_ptr + offs) * 2.0)
    tl.store(sum_ptr + pid,                     # one thread, after the join
             tl.sum(tl.load(out_ptr + base + tl.arange(0, N))))

kernel[(grid,)](x, out, sums, N=4096, BLOCK=128, num_warps=4)
```

`python/examples/cpu_parallel_demo.py` is a runnable version that prints the
threads and CPUs that executed each instance (from the runtime trace).

`cpu.current_cpu()` and `cpu.os_thread_id()` return, inside a kernel, the
CPU (`sched_getcpu()`) and Linux thread id of the thread executing the call.
`python/examples/cpu_thread_placement_demo.py [num_warps] [grid]` uses them
to let every iteration of a `cpu.parallel` loop report where it ran, without
any runtime trace:

```
instance 1: serial part on cpu 4 (tid 464254); loop on 4 threads, CPUs [4, 5, 6, 7], L2 domain(s) ['4-7']
    iteration 0: thread 464254 on cpu 4
    iteration 1: thread 464255 on cpu 5
    ...
```

They are scalar `tt.extern_elementwise` calls of functions exported by the
runtime; triton-shared-opt lowers scalar extern calls with an unknown symbol
to `func.call` (`ExternElementwiseScalarCallConverter`).

* `cpu.parallel(start, end, step, **range_kwargs)` is a `tl.range` whose
  iterations the user declares independent.  Thread `t` of the instance runs
  iterations `t, t + W, t + 2W, ...`; the code after the loop starts when all
  threads are done (fork/join).
* Code outside `cpu.parallel` loops runs on one thread.
* Values defined before the loop (pointers, scalars) are shared by all
  threads, so they access the same buffers.  Tensors created in the loop body
  are private to the thread running the iteration.
* The loop must not carry values from one iteration to the next (compile
  error).  Nested `cpu.parallel` loops: only the outermost one is
  distributed, inner ones run sequentially in each thread.

## `num_warps`

`num_warps` is the number of threads (W) of a program instance:

| `num_warps` | W |
|---|---|
| `> 0` | `num_warps` |
| `0` (default) | `_infer_num_warps()` in `backend/compiler.py` |

The inference is a placeholder: the number of CPUs sharing the L2 cache of the
first available CPU, i.e. one full cluster per instance (4 on the K1).  It is
the place to plug a real inference in.  The resolved value is stored in the
kernel metadata (`kernel.metadata.num_warps`) when the kernel is compiled.

The runtime reserves W cores per instance only if the kernel actually forks a
parallel region (`kernel.metadata.cpu_threads`, W or 1).  A kernel without
`cpu.parallel` loop keeps one core per instance whatever `num_warps` says, so
existing kernels that pass `num_warps` out of GPU habit lose no parallelism.

## Pipeline

```
cpu.parallel loop
  │ Triton code generator (patches/0005): scf.for {cpar.parallel}
  ▼
TTIR ─ triton-shared-opt ─► linalg on tensors, the scf.for keeps its attribute
  │ buddy-opt --cpar-distribute-loops=num-threads=W         (vectorir stage)
  ▼
scf.forall (%t) in (W) {                       // still tensors
  scf.for %iv = lb + t*step to ub step step*W { body }
} {cpar.threads}
  │ buddy-opt --one-shot-bufferize    (forall body = parallel region:
  │                                    tensors written inside get private buffers)
  │ buddy-opt --cpar-forall-to-parallel
  ▼
cpar.parallel num_threads(W) {
^bb0(%tid: i64, %nt: i64):
  scf.for %iv = lb + tid*step to ub step step*nt { body }
  cpar.yield
}
  │ usual lowering to the LLVM dialect, the region stays nested
  │ buddy-opt --cpar-outline-to-runtime                     (llir stage, last)
  ▼
llvm.func internal @kernel__cpar0(%ctx, %tid, %nt) { load captures; body }
llvm.call @__cpar_parallel(@kernel__cpar0, %ctx, W)
```

* **Frontend**: `language/cpu/parallel.py` (installed as
  `triton.language.extra.cpu`).  `patches/0005-range-subclass-loop-annotations.patch`
  makes the Triton code generator accept subclasses of `tl.range` and lets them
  annotate the generated `scf.for` (`_annotate_loop` hook, 10 lines).
* **buddy-mlir** (`midend/include/Dialect/CPar`, `midend/lib/Dialect/CPar`,
  `midend/lib/Conversion/CPar`):
  * `cpar.parallel num_threads(N) { ^bb0(%tid: i64, %nt: i64): ... }`: an SPMD
    region executed by N threads.  It is an automatic allocation scope (stack
    allocations are per thread) and an affine scope; it may hold several blocks
    once control flow is lowered.
  * C++ interface (`CPar/CParTransforms.h`), usable by any pass that wants to
    run code with several threads:
    * `wrapInParallelRegion(rewriter, first, last, N)` wraps an arbitrary range
      of ops of one block into a region (SPMD, the ops use the thread id to
      split the work); fails if a value defined in the range is used after it.
    * `distributeLoop(rewriter, forOp, N)` (tensor level) turns a loop without
      loop-carried values into a one iteration per thread `scf.forall
      {cpar.threads}` whose threads run the iterations round robin;
      `forallToParallel(rewriter, forallOp)` (buffer level) turns it into a
      `cpar.parallel` region.
    * Why the detour through `scf.forall`: bufferizing a sequential loop may
      legally reuse one buffer for a temporary of all iterations (e.g. when
      CSE hoisted the `tensor.empty` out of the loop).  Executed by several
      threads, that buffer would be shared: a data race, and on the K3 a 20x
      slowdown from the cache line bouncing between the cores.  One-Shot
      Bufferize treats the body of an `scf.forall` as a parallel region and
      bufferizes writes to tensors defined outside of it out of place, i.e.
      into buffers private to each thread
      (`tests/Conversion/CPar/private-buffers.mlir`).  `wrapInParallelRegion`
      therefore expects buffer level IR, where the shared memory is explicit.
  * `--cpar-distribute-loops=num-threads=N`: applies `distributeLoop` to the
    loops carrying the `cpar.parallel` attribute (num-threads=1 only drops the
    markers).  Runs before bufferization.
  * `--cpar-forall-to-parallel`: applies `forallToParallel` to the foralls
    created by `--cpar-distribute-loops`.  Runs after bufferization.
  * `--cpar-outline-to-runtime`: runs on the LLVM dialect.  Values used inside a
    region and defined above it are stored into a context struct (constants are
    cloned instead), the region becomes an internal function
    `(ctx, tid, nthreads)`, and the op becomes a call to `__cpar_parallel`.
    The context is allocated in the closest allocation scope, so nested regions
    get one context per thread of the enclosing region.
* **compiler.py**: resolves W, runs `--cpar-distribute-loops=num-threads=W`
  before and `--cpar-forall-to-parallel` after `--one-shot-bufferize`, runs
  `--cpar-outline-to-runtime` after `--reconcile-unrealized-casts`, and sets
  `metadata["cpu_threads"]` from the presence of `call @__cpar_parallel` in the
  LLVM IR.  `pack_metadata` hands it to the launcher.

Runtime ABI (`backend/runtime/triton_cpu_rt.h`):

```c
void __cpar_parallel(void (*fn)(void *ctx, int64_t tid, int64_t nthreads),
                     void *ctx, int64_t nthreads);
```

It calls `fn(ctx, tid, nthreads)` for every `tid` in `[0, nthreads)` and returns
when all calls are done.  It is exported by the runtime library the launcher
links against, so `kernel.o` resolves it at load time.

## Runtime: slots

The runtime keeps one pinned worker per CPU (see document 05).  For a launch
with W threads per instance, every L2 domain is cut into *slots* of
`min(W, domain size)` consecutive workers; the rest of a domain
(`size % slot size`) stays idle.  On the K1 (two clusters of 4):

| W | slot size | slots = concurrent instances | threads in use |
|---|---|---|---|
| 1 | 1 | 8 | 8 |
| 2 | 2 | 4 | 8 |
| 3 | 3 | 2 | 6 |
| 4 | 4 | 2 | 8 |
| 8 | 4 | 2 | 8 |

* The first worker of a slot (leader) executes program instances, taking them
  from the grid's atomic counter.  `TRITON_CPU_NUM_THREADS` still caps the
  number of concurrent instances (slots), not the threads per instance.
* `__cpar_parallel` called by a leader posts the region to the other workers of
  its slot (helpers), runs the logical ids `0, S, 2S, ...` itself (S = threads
  sharing the region = `min(nthreads, slot size)`), helper `m` runs
  `m, m + S, ...`, and the leader waits for the helpers on the slot's latch.
* No oversubscription: if W is larger than a domain, the slot is the whole
  domain and each thread runs several logical ids (W=8 on the K1: 4 threads,
  2 ids each).  The kernel still sees `nthreads = W`.
* Regions called anywhere else (nested regions, a launch that runs on the
  calling thread, calls outside a launch) run the ids one after the other on
  the calling thread.
* The total number of threads never changes: the pool is created once with one
  worker per CPU, W only changes how workers are grouped.
* Because every worker is pinned to its CPU and a slot never spans two domains,
  the threads of an instance cannot run outside their cluster.

## Trace

With `TRITON_CPU_TRACE_FILE=<file>` the runtime writes

* `<file>`: `instance,start_ns,end_ns,thread_id,cpu,os_tid`, one row per program
  instance; `thread_id` is the slot rank, `os_tid` the leader's thread id.
* `<file>.threads`: `instance,region,tid,nthreads,os_tid,cpu,start_ns,end_ns`,
  one row per call `fn(ctx, tid, nthreads)`; `cpu` is `sched_getcpu()` at the
  start of the call.  Empty (header only) for kernels without region.

## Constraints

* The iterations of a `cpu.parallel` loop must be independent; nothing checks
  it.  Writes of different iterations to the same element race.
* No loop-carried values (accumulators); reductions across threads are not
  supported yet.  Reduce per thread into memory and combine after the loop.
* No synchronization inside a region (no barrier); the only synchronization
  point is the end of the region.
* W is a compile-time value: changing `num_warps` recompiles the kernel.  The
  inferred value depends on the topology of the compiling machine, which is
  also the executing machine for the JIT.
* The `cpar.parallel` attribute has to survive triton-shared-opt.  The
  structural type conversions and the bufferization keep discardable
  attributes; a loop that a pass recreates without copying its attributes
  silently runs on one thread (visible in the trace: no `.threads` rows).

## Verification

`python/examples/test_program_instance_threads.py`:

| Test | Covers |
|---|---|
| `test_num_warps_forks_threads` | AC1: W ∈ {2, cluster size}: every instance runs tids 0..W-1 on W OS threads, tid 0 on the instance's thread; the serial sum after the loop reads all values written by the threads (join). |
| `test_num_warps_zero_infers` | AC1: num_warps=0 resolves to the inferred W. |
| `test_kernel_reports_one_cluster` | AC2, measured by the kernel: every iteration stores `cpu.current_cpu()` / `cpu.os_thread_id()`; per instance all CPUs lie in one L2 domain (sysfs), W distinct threads, iteration i always on thread i % W. |
| `test_threads_stay_in_one_cluster` | AC2: the CPUs measured by `sched_getcpu()` in every thread of an instance belong to one L2 domain (checked against the runtime's topology and against sysfs directly), and each thread's affinity mask (`os.sched_getaffinity(tid)`) is one CPU of that domain. |
| `test_warps_reduce_instance_parallelism` | AC4: for W ∈ {1, 2, cluster size} the measured maximum number of concurrent instances is bounded by the slot count and decreases with W; the threads used never exceed the CPU count and the process thread count does not grow with W. |
| `test_parallel_and_serial_parts` | AC5: the loop runs on W threads, the code around it on one; a kernel without `cpu.parallel` gets `cpu_threads = 1`. |
| `test_threads_do_not_share_temporaries` | Results with W ∈ {2, cluster size} are bit-identical to W=1 for a kernel whose loop body computes temporaries (detects threads sharing a temporary buffer). |
| `test_region_speedup` | One instance, wall clock W=1 vs. W = cluster size. |

buddy-mlir lit tests: `tests/Dialect/CPar/{ops,invalid}.mlir`,
`tests/Conversion/CPar/{distribute-loops,forall-to-parallel,private-buffers,outline-to-runtime}.mlir`.

Run on the target:

```sh
source env.sh
cd triton-riscv/python/examples
pytest -q test_program_instance_threads.py test_program_instance_parallel.py -s
python cpu_parallel_demo.py 4
```

## Measured on the SpacemiT K3

K3: 8 X100 cores visible to processes (`cpu0-7`, 2150 MHz) in two L2 domains
`0-3` and `4-7`; the 8 A100 cores (`cpu8-15`) are online but outside the
process affinity mask, so the runtime does not use them.

```
$ pytest -q test_program_instance_threads.py test_program_instance_parallel.py -s
num_warps=2: 3 instances x 2 OS threads
num_warps=4: 3 instances x 4 OS threads
num_warps=0 -> 4 threads per instance
4 instances x 4 threads, each instance inside one L2 domain
num_warps=1: slots=8 max concurrent instances=8 threads used=8 process threads=17
num_warps=2: slots=4 max concurrent instances=4 threads used=8 process threads=17
num_warps=4: slots=2 max concurrent instances=2 threads used=8 process threads=17
instance 0: parallel part 4 threads, serial part 1 thread
instance 1: parallel part 4 threads, serial part 1 thread
one instance: num_warps=1 48.9ms, num_warps=4 12.7ms, speedup 3.84x
serial=312.40ms parallel=39.63ms speedup=7.88x (8 cpus)
12 passed in 19.47s

$ python cpu_parallel_demo.py 8
num_warps=8 -> 8 threads per instance (cpu_threads=8); results OK
  instance 0: tids [0, 1, 2, 3, 4, 5, 6, 7] on 4 OS threads, CPUs [0, 1, 2, 3]
  ...
```

`num_warps=8` exceeds the cluster size: each instance still sees 8 logical
threads, executed by the 4 threads of one cluster (no oversubscription).
The 77 existing example tests pass.

TODO: the same on the K1.
