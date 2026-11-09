# Program instance level parallelism

The CPU backend used to traverse the launch grid with a plain three level
`for` loop in the generated launcher, i.e. every program instance was executed
one after the other.  The grid is now executed in parallel by a small CPU
runtime that owns a pool of worker threads; this document describes the
mechanism, the knobs it exposes and how the behaviour is verified.

## Where it lives

| File | Role |
|---|---|
| `backend/runtime/triton_cpu_rt.h` | C ABI between launchers and the runtime (`tcrt_launch`, topology queries). |
| `backend/runtime/triton_cpu_rt.cpp` | Topology detection, worker pool, grid scheduling, trace. |
| `backend/driver.py` | Builds the runtime library, generates and builds the per-signature launcher. |

The generated launcher no longer contains any scheduling code.  It packs the
kernel arguments into a struct, provides a per instance entry point
`_instance(args, x, y, z)` and hands both to `tcrt_launch()`.

No Triton compiler change is involved: the runtime and the launcher are built
at run time on the machine that executes the kernels.  A change to the
parallelism only requires syncing `backend/driver.py` and `backend/runtime/`
to the target.

## One runtime per process

Triton builds one launcher extension per kernel signature.  If the worker
pool lived in the launcher, every signature would come with its own pool.  The
runtime is therefore a separate shared library:

* `driver.py` compiles `triton_cpu_rt.cpp` on first use
  (`g++ -O2 -std=c++17 -fPIC -shared -pthread`, plus the sanitizer flags when
  `TRITON_SHARED_SANITIZER_TYPE` is set) and stores it in the Triton cache.
* The file name and the soname are `libtriton_cpu_rt.<hash>.so`, where the hash
  covers the sources and the compile command.  The dynamic loader identifies
  libraries by soname, so two different builds can never be merged into one.
* Launchers link against that file with an rpath to its cache directory, and
  their cache key includes the runtime hash.

The result is a single pool per process, however many kernels are launched.
On Windows (not a target platform) there is no shared library: the runtime is
compiled into every launcher.

## Worker pool

* **Topology.** The runtime takes the CPUs available to the process
  (`sched_getaffinity`, so `taskset`/cgroup restrictions are honoured) and
  groups them into *domains* of CPUs that share an L2 cache:
  `cpuN/cache/index*/shared_cpu_list` of the level 2 cache, falling back to
  `cpuN/topology/cluster_cpus_list`, falling back to a single domain.
  `TRITON_CPU_CLUSTERS="0-3;4-7"` overrides the detection.  On the K1 both
  sysfs sources give the two clusters `0-3` and `4-7`.
* **Workers.** The first launch that needs more than one worker creates one
  thread per available CPU.  Every worker is pinned to its CPU when it is
  created (`pthread_attr_setaffinity_np`) and never migrates afterwards.
  Asynchronous signals are blocked in workers, so they keep going to the
  application threads (the Python main thread).
* **The calling thread is not a worker.** It is never pinned and only waits for
  the launch to finish, so launching a kernel does not change the scheduling of
  the Python thread or of other libraries running on it (e.g. torch).
* **Scheduling.** The caller posts the grid to P workers.  Each worker grabs the
  next program instance from a shared atomic counter, which is the equivalent
  of `schedule(dynamic, 1)`: program instances do not all cost the same, and the
  K1's two clusters do not run at the same speed, so a static split would leave
  a long tail.  The flattening of the grid is unchanged (`x` outer, `z` inner).
* **Choice of workers.** Workers are taken round robin over the domains, so a
  launch with fewer instances than CPUs spreads over all clusters.
* **Waiting.** Idle workers wait on a per-worker mailbox and the caller waits on
  a completion latch.  Both spin for `TRITON_CPU_SPIN` iterations and then
  sleep on a futex.

## AI cores (SpacemiT K3)

The K3 has 8 X100 cores (`cpu0-7`) and 8 A100 "AI" cores (`cpu8-15`, two L2
domains `8-11` and `12-15`).  The A100 cores are outside the affinity of
ordinary threads: `sched_setaffinity` to them fails with `EINVAL`.  A thread
gets access to them by writing its own thread id to `/proc/set_ai_thread`
(writable by every user).  Measured on kernel 6.18.3:

* the thread is moved to the A100 cores at once; its affinity becomes `8-15`
  and it can then be pinned to a single A100 core;
* the move is one way: the thread can no longer be pinned to an X100 core;
* only the writing thread is affected; threads it creates inherit its
  affinity, other threads of the process are not touched.

By default, when `/proc/set_ai_thread` exists, the pool runs on the AI cores
only, so that kernels can later use A100-only instructions and the A100 TCM:

* **Topology.** A short-lived thread moves itself to the AI cores and reads
  its affinity; that set (not a hard-coded `8-15`) replaces the CPUs of the
  process.  Domains are detected from sysfs as usual.
* **Workers.** Another short-lived thread moves itself to the AI cores and
  creates the workers.  They inherit the access and are pinned to their AI
  core at creation (`pthread_attr_setaffinity_np`), so they never execute
  anything on an X100 core and no running thread is ever migrated (moving a
  running thread between cores with different VLEN, 256 vs 1024 bit, is a
  known source of crashes).  A worker that cannot be pinned is left out
  instead of running unpinned.
* **No kernel code on the calling thread.** The `P == 1` shortcut below is
  disabled: a launch with one instance is posted to one worker.  The calling
  thread is never moved, so its affinity stays `0-7`.  The cost is about
  1 µs per launch (0.99 µs instead of 0.23 µs for an empty grid=1 launch).
* **Fallback.** If moving a thread fails, the runtime warns once and uses the
  CPUs of the process.  `TRITON_CPU_AI_CORES=0` selects that behaviour
  explicitly.  Without `/proc/set_ai_thread` (K1, x86, qemu) nothing changes.
* `tcrt_ai_cores()` reports the mode; `compiler.py::_ai_core_cpus()` performs
  the same probe so that the `num_warps=0` inference uses the AI domains.

## Number of concurrent program instances

`P = min(available CPUs, TRITON_CPU_NUM_THREADS if > 0, number of instances)`.

If `P == 1` (single CPU, a single instance, or `TRITON_CPU_NUM_THREADS=1`), the
grid runs on the calling thread and the pool is not involved at all, except on
the AI cores, where it runs on one worker (see above).

`TRITON_CPU_NUM_THREADS` is now capped at the number of available CPUs: there
is exactly one worker per CPU, so asking for more cannot create more
concurrency.

## Knobs

| Environment variable | Meaning |
|---|---|
| `TRITON_CPU_NUM_THREADS` | Maximum number of program instances executed concurrently. `0`/unset means "one per available CPU". Used by the tests to obtain a serial baseline. Read at every launch. |
| `TRITON_CPU_TRACE_FILE` | If set, the runtime writes one CSV line per program instance: `instance,start_ns,end_ns,thread_id,cpu,os_tid` (and `<file>.threads` for parallel regions, see document 06). Debug/test facility only. Read at every launch. |
| `TRITON_CPU_CLUSTERS` | Overrides the domain detection, e.g. `0-3;4-7`. CPUs it does not mention form their own domains. Read when the pool is created. |
| `TRITON_CPU_SPIN` | Spin iterations before a waiting thread sleeps (default `4000`). Higher values lower the wake-up latency of back-to-back launches and burn more CPU while idle. Read when the pool is created. |
| `TRITON_CPU_AI_CORES` | `0` keeps the pool off the SpacemiT AI cores (A100) and on the CPUs of the process. Any other value, or unset: use the AI cores when `/proc/set_ai_thread` exists. Read when the pool is created. |
| `TRITON_CPU_TCM`, `TRITON_CPU_TCM_LIMIT` | TCM for `cpu.alloc` buffers, see [07-CPU-Alloc-TCM.md](07-CPU-Alloc-TCM.md). |

In the trace, `thread_id` is the rank of the worker within the launch (`0` for
the calling thread when `P == 1`) and `os_tid` its Linux thread id.  `cpu` is `sched_getcpu()` at instance start,
or `-1` on non-Linux hosts.  `start_ns`/`end_ns` come from
`std::chrono::steady_clock`.  Every instance writes its own slot, so recording
needs no synchronization, and the file is written after the grid finished.

OpenMP is no longer involved: `OMP_NUM_THREADS`, `OMP_PROC_BIND`, `OMP_PLACES`
and friends have no effect on the launcher.

## Process lifecycle

* `fork()`: in the child the pool is forgotten (its threads do not exist
  there) and rebuilt on first use.  `pthread_atfork` also makes sure no launch
  is in flight while a process forks.
* Exit: workers are detached threads sleeping on a futex; they do not delay
  `exit()`.  Workers holding a TCM block release it from an `atexit` handler
  (see [07-CPU-Alloc-TCM.md](07-CPU-Alloc-TCM.md)).
* Launches from several application threads are serialized by a lock inside
  the runtime.  The generated launcher currently holds the GIL during the
  launch anyway.

## Semantics

Program instances are independent by definition of the Triton programming
model (there is no synchronization between them), so executing them in any
order, concurrently, produces the same result as the previous serial traversal.

All program instances still share the process' main memory.  Running several
threads per program instance on one cluster (`tcrt_launch_desc::warps`,
`num_warps`) builds on the domains and the pinned workers introduced here, see
[06-Program-Instance-Threads.md](06-Program-Instance-Threads.md).  With
`warps = 1` (the default for kernels without `cpu.parallel` loop) the behaviour
described in this document is unchanged.

## Verification

`python/examples/test_program_instance_parallel.py` covers this feature:

* `test_grid_parallel_coverage`: every instance runs exactly once and gets the
  program ids of its grid slot, both for 1D and 3D grids (this also pins the
  flattening of `x`/`y`/`z`).
* `test_grid_parallel_overlap`: a busy kernel runs on `2 * ncpu` instances while
  `TRITON_CPU_TRACE_FILE` records the instance timeline.  The test asserts three
  things:
  * more than one worker was used;
  * at least one pair of instances interleaves (`a.start < b.start < a.end < b.end`),
    which is impossible for a serial traversal;
  * the achieved concurrency `sum(end - start) / span` is above 1.5.
* `test_grid_workers_pinned`: every worker runs all of its instances on a
  single CPU, and no two workers share a CPU.
* `test_grid_parallel_serial_override`: `TRITON_CPU_NUM_THREADS=1` yields one
  thread, no interleaving, and still correct results.
* `test_grid_parallel_speedup`: wall-clock speedup of the parallel run over the
  serial baseline (skipped on machines with a single CPU).

Run it on the target with:

```sh
source env.sh
cd triton-riscv/python/examples
pytest -q test_program_instance_parallel.py -s
```

## Measured on the SpacemiT K1

K1: 8 cores in two clusters (`cpu0-3` → cluster 0, `cpu4-7` → cluster 1, each
cluster with its own L2), `nproc = 8`.

TODO: re-run on the K1 with the runtime based implementation and record the
output of `pytest -q test_program_instance_parallel.py -s` here.  The numbers
previously recorded here were obtained with the OpenMP based launcher
(6.94x speedup, concurrency 7.51 on 16 instances).
