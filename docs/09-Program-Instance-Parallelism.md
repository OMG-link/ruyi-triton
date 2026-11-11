# Program instance parallelism

A Triton launch grid is a set of program instances without synchronization
between them. The CPU backend executes them concurrently on a pool of worker
threads owned by the CPU runtime, `backend/runtime/triton_cpu_rt.{h,cpp}`.
[Threads per program instance](10-Program-Instance-Threads.md) builds on it.

## Board configurations

Which CPUs run kernels, and how, is described by a board configuration
(`backend/board.py`). The runtime itself knows nothing about the board.

| Configuration | Compute CPUs, by L2 domain | IME | SPM | Reaching the CPUs |
|---|---|---|---|---|
| `k1-x60` | `0-3; 4-7` | none | none | plain affinity |
| `k1-a60` | `0-3` | IME1 | 512 KiB, 128 KiB per CPU | plain affinity |
| `k3-x100` | `0-3; 4-7` | IME1 | none | plain affinity |
| `k3-a100` | `8-11; 12-15` | IME2 | 3 MiB, 384 KiB per CPU | `/proc/set_ai_thread` |
| `generic` | the CPUs of the process, grouped by the L2 caches sysfs reports | | none | plain affinity |

A kernel's configuration is fixed when it is compiled, from the compile
target: `spacemit-a100` gives `k3-a100`, `spacemit-x100` gives `k3-x100`,
`spacemit-x60` gives `k1-a60` (LLVM has one name for both K1 core types; K1
kernels run on the A60 cores), anything else `generic`. `TRITON_RISCV_BOARD`
names a configuration explicitly. The configuration is part of the kernel's
metadata (`board_config`).

A process runs all its kernels with one configuration. The first launch
configures the runtime; a kernel compiled for another configuration fails to
launch with a `RuntimeError`.

On the K3, LLVM detects the host CPU of the core the compiling thread runs
on. Compile with `TRITON_RISCV_MCPU=spacemit-a100` (or from a thread already
on the A100 cores) so that IME2 kernels get the `k3-a100` configuration.

`TRITON_CPAR_CPUS` replaces the domains of the configuration. Domains are
separated by `;`, CPUs inside one by `,`, ranges by `-`:

| `TRITON_CPAR_CPUS` | Effect |
|---|---|
| `8` | one CPU: program instances run one after the other (a serial baseline) |
| `0-3;4-7` on the K3 | the X100 cores instead of the A100 cores |
| `0-3;4-7` on a generic host | two domains of four CPUs, whatever sysfs says |

A CPU keeps its scratchpad share; CPUs that have none get none. Threads enter
the AI cores only if every listed CPU belongs to the configuration's own
CPUs, because a thread moved there can no longer reach other cores.

## Runtime

* **Pool.** One worker thread per compute CPU, pinned to it when it is
  created and never migrated. A CPU a worker cannot be pinned to is left out
  with a warning. On `k3-a100` a short lived thread writes its id to
  `/proc/set_ai_thread` and creates the workers, which inherit access to the
  A100 cores; the calling thread is never moved.
* **Kernel code runs only on workers**, never on the calling thread, so it
  never runs on a CPU outside the configuration: an IME kernel on an X60 or
  X100 core would be a SIGILL, a TCM access on a K1 X60 core a SIGSEGV. The
  cost is about 1 µs per launch.
* **Slots.** For a launch with W threads per program instance the compute
  CPUs are cut into slots of W consecutive workers. If the rest of the
  current domain is too small for another slot, the next slot starts at the
  next domain and the rest stays idle. A slot larger than a domain starts at
  a domain boundary and spans the following domains, with a warning, since
  its threads then do not share one domain's cache. A slot never has more CPUs than the
  configuration; the kernel still sees W threads. Slots are handed out round
  robin over the domains, so a small grid spreads over the domains.
* **Launch.** The calling thread posts the grid to the leaders (first
  workers) of up to as many slots as there are program instances. Leaders
  take instances from an atomic counter, one at a time, and count down a
  latch when the grid is drained. Waiting threads spin briefly and then sleep
  on a futex.
* **Scratchpad.** When the runtime is configured, its first worker acquires
  the board's whole scratchpad for the process and holds it until the process
  exits normally (`atexit`; the K3 driver requires the acquiring thread to
  release). A process killed while holding it leaves stale records on the
  K3: `python -m triton.backends.triton_shared.spm_tool status|release-stale`.
  Every slot gets the shares of its CPUs: on `k3-a100` an instance on 8-11
  gets the first half of the TCM, one on 12-15 the second half. A kernel's
  `tl.dev.alloc(placement="spm")` buffers live in its slot's share; a launch
  whose kernel needs more, or runs on CPUs without a share, or finds the
  scratchpad held by another process, raises `SpmOutOfMemoryError` before
  any instance runs. Another process trying to acquire the scratchpad while
  this one holds it is refused; it is never taken away.

The runtime is a shared library built with `g++` on first use on the machine
that runs the kernels, for the configuration's scratchpad driver
(`include/spm/`), and cached like other Triton artifacts. It is loaded once per
process with `RTLD_GLOBAL` (`backend/cpu_runtime.py`); every launcher links
against it, so a process has a single pool. A process that forks must not
launch kernels in the child: the child has no workers.

## Launcher

The generated launcher (`backend/driver.py`) packs the kernel arguments into
a struct and calls

```c
int tcrt_launch(const tcrt_launch_desc *desc);
// desc: grid, warps (metadata cpu_threads), spm_bytes, spm_align,
//       fn(args, spm, x, y, z), args, trace_path
```

`fn` runs one program instance; `spm` is its slot's scratchpad share.

## Trace

With `TRITON_CPAR_TRACE_FILE=<file>` the runtime writes `<file>` with one row
per program instance (`instance,start_ns,end_ns,thread_id,cpu,os_tid`;
`thread_id` is the slot rank) and `<file>.threads` with one row per call of
a parallel region body (see document 10).

## Semantics

Program instances now run concurrently, as on a GPU. A kernel whose
instances write the same element (e.g. a scatter with duplicate indices)
gives a result that depends on timing; under a serial grid it happened to be
deterministic.

## Tests

`python/examples/test_program_instance_parallel.py`: grid coverage, overlap
of instance intervals, the single CPU override, the slot layout rule, and
the `tl.dev.parallel` tests of document 10. `python/examples/test_alloc.py`
covers the scratchpad.
