# Program instance level parallelism

The CPU backend used to traverse the launch grid with a plain three level
`for` loop in the generated launcher, i.e. every program instance was executed
one after the other.  The launcher now executes the grid in parallel; this
document describes the mechanism, the knobs it exposes and how the behaviour is
verified.

## Where it lives

Everything is in `backend/driver.py`:

* `_generate_launcher()` emits the C++ launcher source, including `_launch()`.
* `_launch()` traverses `gridX * gridY * gridZ` program instances with an
  OpenMP `parallel for collapse(3)` loop.
* `compile_module()` compiles the launcher with `g++ -fopenmp` on the machine
  that executes the kernel.

No Triton compiler change is involved: `num_warps` and the metadata are already
available at the Python level (`CPUOptions`, `pack_metadata()`), and the
launcher is generated and compiled at run time.  Consequently a change to the
parallelism only requires syncing `backend/driver.py` to the target.

## Number of concurrent program instances

`_triton_cpu_num_worker_threads()` decides how many program instances run at
the same time:

1. `TRITON_CPU_NUM_THREADS` if set to a value `> 0`,
2. otherwise `omp_get_num_procs()` (which follows the CPU affinity mask),
3. otherwise `sched_getaffinity()` on Linux,
4. and finally the number of program instances itself (a grid with fewer
   instances than CPUs never starts more workers than it has work for).

The result is passed to the loop as an explicit `num_threads(...)` clause, so
`OMP_NUM_THREADS` has no effect on the number of workers.  `if (nthreads > 1)`
serializes the region for small launches (including `grid == 1`, which keeps the
single instance case completely free of any OpenMP team setup).

`schedule(dynamic, 1)` is used on purpose: program instances do not all cost the
same amount of time, and the SpacemiT K1 has two clusters with different cores,
so a static split would leave a long tail.

## Knobs

| Environment variable | Meaning |
|---|---|
| `TRITON_CPU_NUM_THREADS` | Maximum number of program instances executed concurrently. `0`/unset means "one per available CPU". Used by the tests to obtain a serial baseline. |
| `TRITON_CPU_TRACE_FILE` | If set, the launcher writes one CSV line per program instance: `instance,start_ns,end_ns,thread_id,cpu`. Debug/test facility only: when unset the launcher does a single `getenv()`. |

`cpu` is `-1` on non-Linux hosts (no `sched_getcpu()`).

The trace is written after the loop finishes, into the slot owned by each
instance, so recording needs no synchronization.  `thread_id` is the OpenMP
thread number of the worker that executed the instance, `start_ns`/`end_ns` come
from `std::chrono::steady_clock`.

## Toolchains without OpenMP

The parallel loop is guarded with `#if defined(_OPENMP)`.  If the launcher
compilation with `-fopenmp` fails (for example Apple clang without `libomp`),
`compile_module()` retries without the flag and prints a warning: the grid is
then traversed serially, everything else keeps working.

## Interplay with OpenMP environment variables

* `TRITON_CPU_NUM_THREADS` wins over `OMP_NUM_THREADS` because the worker count
  is passed as an explicit `num_threads(...)` clause.
* `OMP_PROC_BIND` / `OMP_PLACES` / `OMP_WAIT_POLICY` still influence the
  libgomp runtime, i.e. how the workers are placed and how they wait.
* A Triton launch performed from inside a user OpenMP parallel region is a
  nested region.  Nested parallelism is disabled by default, so such a launch
  is executed serially unless the application sets
  `OMP_MAX_ACTIVE_LEVELS`/`omp_set_max_active_levels()`.

## Semantics

Program instances are independent by definition of the Triton programming
model (there is no synchronization between them), so executing them in any
order, concurrently, produces the same result as the previous serial traversal.
The traversal order of the flattened loop is unchanged (`x` outer, `z` inner).

Two limitations are worth knowing:

* `num_warps` is still ignored by the runtime.  Program instances run on a
  single thread each; per-instance warp threads (and the cluster affinity that
  comes with them) are the next step.  The worker budget above is where that
  will plug in: the number of concurrent instances has to shrink accordingly.
* All program instances still share the process' main memory.  Nothing in this
  change tries to place them on the same cluster / L2 yet.

## Verification

`python/examples/test_program_instance_parallel.py` covers this feature:

* `test_grid_parallel_coverage` – every instance runs exactly once and gets the
  program ids of its grid slot, both for 1D and 3D grids (this also pins the
  flattening of `x`/`y`/`z`).
* `test_grid_parallel_overlap` – a busy kernel runs on `2 * ncpu` instances while
  `TRITON_CPU_TRACE_FILE` records the instance timeline.  The test asserts that
  (a) more than one launcher thread was used, (b) at least one pair of
  instances interleaves (`a.start < b.start < a.end < b.end`) - impossible for a
  serial traversal - and (c) the achieved concurrency `sum(end - start) / span`
  is above 1.5.
* `test_grid_parallel_serial_override` – `TRITON_CPU_NUM_THREADS=1` yields one
  thread, no interleaving, and still correct results.
* `test_grid_parallel_speedup` – wall-clock speedup of the parallel run over the
  serial baseline (skipped on machines with a single CPU).

Run it on the target with:

```sh
source env.sh
cd triton-riscv/python/examples
pytest -q test_program_instance_parallel.py -s
```

## Measured on the SpacemiT K1

K1: 8 cores in two clusters (`cpu0-3` → cluster 0, `cpu4-7` → cluster 1),
`nproc = 8`, `g++ 13.2` with `libgomp` available.

```
$ pytest -q test_program_instance_parallel.py -s
...serial=279.28ms parallel=40.23ms speedup=6.94x (8 cpus)
.
4 passed in 19.94s
```

The trace of the busy kernel on 16 program instances shows the expected
picture (one instance per `start_ns`/`end_ns` interval):

```
cpus=8 instances=16 iterations=10968
parallel: threads=[0, 1, 2, 3, 4, 5, 6, 7]
parallel: cpus=[0, 1, 2, 3, 4, 5, 6, 7]
parallel: concurrency=7.51 interleaved_pairs=58
serial:   threads=[0] cpus=[0]
serial:   concurrency=1.00 interleaved_pairs=0
```

i.e. the 8 workers are spread over all cores of both clusters, their intervals
interleave heavily (`58` interleaved pairs - impossible for a serial traversal)
and `sum(duration) / span` reaches `7.51` of the theoretical `8`.  The
`TRITON_CPU_NUM_THREADS=1` baseline stays on a single thread and yields no
interleaving at all.

Wall clock of the complete test file: ~20 s on the K1 (dominated by kernel
compilation and calibration).

Existing kernels are unaffected by the parallel traversal: `test_vec_add.py`,
`test_reduce.py`, `test_softmax.py` (which passes `num_warps`),
`test_mask_loop_iter_arg.py`, `test_scalar_store.py`, `test_matmul.py`,
`test_mm.py`, `test_mask.py` and `test_addptr.py` (77 tests) pass on the K1 with
the parallel launcher.
