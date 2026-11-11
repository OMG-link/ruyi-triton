//===- triton_cpu_rt.h - triton-riscv CPU runtime ABI -----------*- C -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// Process wide runtime of the triton-riscv CPU backend.
//
// The runtime owns a pool of worker threads, one per compute CPU of the
// board configuration (backend/board.py), each pinned to its CPU when it is
// created. Generated launchers hand a whole launch grid to tcrt_launch(),
// which executes the program instances on the pool. Compiled kernels call
// __cpar_parallel() to run a code region with several threads inside one
// program instance.
//
// The board's scratchpad (SPM) is acquired once, in full, when the runtime
// is configured, and held until the process exits. Every slot (the CPUs that
// run one program instance) gets the part of it that belongs to its CPUs.
//
// The runtime is built once per board into its own shared library (see
// driver.py), so that every kernel launcher shares the same pool.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CPU_RT_H
#define TRITON_CPU_RT_H

#include <stdint.h>

#define TCRT_API __attribute__((visibility("default")))

#ifdef __cplusplus
extern "C" {
#endif

#define TCRT_ABI_VERSION 2

//===----------------------------------------------------------------------===//
// Configuration
//===----------------------------------------------------------------------===//

// The compute CPUs of a board configuration, in order, partitioned into
// domains of consecutive entries (CPUs sharing an L2 cache).
typedef struct {
  int abi_version; // TCRT_ABI_VERSION
  int ncpus;
  const int *cpus; // ncpus entries
  int ndomains;
  const int *domain_sizes; // ndomains entries, summing to ncpus
  // Nonzero: the CPUs are reachable only by threads that wrote their id to
  // /proc/set_ai_thread (the A100 cores of the SpacemiT K3). Workers are then
  // created by such a thread, and kernel code never runs on the caller.
  int enter_ai_cores;
  // SPM bytes that belong to each CPU listed in `cpus`, in the same order,
  // starting at `spm_first_cpu`'s share at offset 0. 0 without SPM.
  int64_t spm_bytes_per_cpu;
  int spm_first_cpu;
} tcrt_config;

// Status of the scratchpad, returned by tcrt_configure / tcrt_launch.
// The values match SPM_* in include/spm/spm_common.h.
#define TCRT_OK 0
#define TCRT_ERROR_INVALID (-1)     // invalid descriptor or configuration
#define TCRT_ERROR_RECONFIGURE (-2) // a different configuration is active
#define TCRT_ERROR_SPM (-3)         // see tcrt_spm_status()

// Configures the runtime: creates the pool and acquires the scratchpad. The
// first call wins; a later call with the same configuration is a no-op, a
// different one returns TCRT_ERROR_RECONFIGURE. A scratchpad that cannot be
// acquired does not fail the call; launches that need it fail instead.
TCRT_API int tcrt_configure(const tcrt_config *config);

// Why the scratchpad is not available (an SPM_ERR_* code, SPM_OK if it is),
// plus the driver's details of the failure, for error reports.
typedef struct {
  int code;
  int busy_block;     // SPM_ERR_OCCUPIED: a block found held, or -1
  int sys_errno;      // SPM_ERR_DEVICE: errno of the failing call
  int64_t bytes;      // size of the held region
  int64_t slot_bytes; // TCRT_ERROR_SPM from a launch: the slot's share
  int slot_cpu;       // TCRT_ERROR_SPM from a launch: the slot's first CPU
} tcrt_spm_status_t;
TCRT_API void tcrt_spm_status(tcrt_spm_status_t *status);

//===----------------------------------------------------------------------===//
// Launcher ABI
//===----------------------------------------------------------------------===//

// Executes the program instance with program ids (x, y, z). `spm` is the
// scratchpad region of the slot running it (NULL without scratchpad).
typedef void (*tcrt_instance_fn)(void *args, void *spm, int x, int y, int z);

typedef struct {
  int abi_version;        // TCRT_ABI_VERSION
  int grid[3];            // launch grid; any entry <= 0 means "nothing to do"
  int warps;              // threads per program instance (values < 1 mean 1)
  int64_t spm_bytes;      // scratchpad bytes per program instance, 0 if none
  int64_t spm_align;      // required alignment of the region
  tcrt_instance_fn fn;    // called once per program instance
  void *args;             // passed through to fn
  const char *trace_path; // optional CSV trace of the launch, NULL to disable
} tcrt_launch_desc;

// Runs every program instance of the grid exactly once and returns when all
// of them are finished. Requires tcrt_configure().
//
// The compute CPUs are cut into slots of `warps` consecutive CPUs. A slot
// stays inside one domain when `warps` fits in it: if the rest of the
// current domain is too small for another slot, the next slot starts at the
// next domain and the rest stays idle. A slot larger than a domain starts at
// a domain boundary and spans the following domains (with a warning). A
// slot never has more CPUs than the configuration; the kernel still sees
// `warps` threads. Slots do not overlap, so the number of program instances
// running concurrently is the number of slots, and the total number of
// threads never exceeds the number of compute CPUs.
//
// Returns TCRT_OK, TCRT_ERROR_SPM if the kernel needs scratchpad the slots
// cannot provide (nothing was executed), or TCRT_ERROR_INVALID.
TCRT_API int tcrt_launch(const tcrt_launch_desc *desc);

//===----------------------------------------------------------------------===//
// Kernel ABI
//===----------------------------------------------------------------------===//

// Body of a parallel region: called once per logical thread id `tid` in
// [0, nthreads).
typedef void (*cpar_region_fn)(void *ctx, int64_t tid, int64_t nthreads);

// Runs fn(ctx, tid, nthreads) for every tid in [0, nthreads) and returns when
// all calls have finished (fork/join). Called from compiled kernels, see the
// `cpar` dialect in buddy-mlir.
//
// Inside a program instance of a launch with warps > 1, the calls are spread
// over the CPUs of the instance's slot: the calling thread runs tid 0 and the
// other threads of the slot run the remaining ids round robin. Everywhere
// else (single CPU slot, nested regions, calls outside of tcrt_launch) the
// calls are made one after the other on the calling thread.
TCRT_API void __cpar_parallel(cpar_region_fn fn, void *ctx, int64_t nthreads);

// CPU the calling thread is running on (sched_getcpu), -1 if unknown.
TCRT_API int32_t tcrt_current_cpu(void);
// Linux thread id (gettid) of the calling thread.
TCRT_API int32_t tcrt_os_thread_id(void);

//===----------------------------------------------------------------------===//
// Introspection (tests and debugging)
//===----------------------------------------------------------------------===//

// Number of slots, i.e. the maximum number of concurrent program instances,
// of a launch with `warps` threads per program instance. 0 if unconfigured.
TCRT_API int tcrt_num_slots(int warps);
// Writes up to `cap` CPU ids of slot `slot` of a launch with `warps` threads
// per instance into `cpus`; returns the slot size or -1.
TCRT_API int tcrt_slot_cpus(int warps, int slot, int *cpus, int cap);
// Number of worker threads of the pool (0 if unconfigured).
TCRT_API int tcrt_num_workers(void);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRITON_CPU_RT_H
