//===- triton_cpu_rt.h - triton-riscv CPU runtime ABI -----------*- C -*-===//
//
// Process wide runtime of the triton-riscv CPU backend.
//
// The runtime owns a pool of worker threads, one per CPU available to the
// process, each pinned to its CPU when it is created.  On the SpacemiT K3
// (/proc/set_ai_thread present) the pool uses the AI cores (A100) instead, and
// all kernel code runs on them; TRITON_CPU_AI_CORES=0 opts out.  Generated
// launchers hand
// a whole launch grid to tcrt_launch(), which executes the program instances
// on the pool.  Compiled kernels call __cpar_parallel() to run a code region
// with several threads inside one program instance.
//
// The runtime is built once per process into its own shared library (see
// driver.py) so that every kernel launcher shares the same pool.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CPU_RT_H
#define TRITON_CPU_RT_H

#include <stdint.h>

#if defined(_WIN32)
#define TCRT_API
#else
#define TCRT_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define TCRT_ABI_VERSION 1

//===----------------------------------------------------------------------===//
// Launcher ABI
//===----------------------------------------------------------------------===//

// Executes the program instance with program ids (x, y, z).
typedef void (*tcrt_instance_fn)(void *args, int x, int y, int z);

typedef struct {
  int abi_version;        // TCRT_ABI_VERSION
  int grid[3];            // launch grid; any entry <= 0 means "nothing to do"
  int warps;              // threads per program instance (values < 1 mean 1)
  tcrt_instance_fn fn;    // called once per program instance
  void *args;             // passed through to fn
  const char *trace_path; // optional CSV trace of the launch, NULL to disable
} tcrt_launch_desc;

// Runs every program instance of the grid exactly once and returns when all of
// them are finished.
//
// Every program instance is given a "slot": min(warps, domain size) CPUs of a
// single domain (CPUs sharing an L2 cache).  Slots do not overlap, so the
// number of program instances running concurrently is the number of slots
// (capped by TRITON_CPU_NUM_THREADS and the grid size), and the total number of
// threads never exceeds the number of CPUs.
//
// Returns 0 on success, TCRT_ERROR_NO_TCM if TRITON_CPU_TCM=require and a
// cpu.alloc buffer could not be put in the TCM (the launch was still executed,
// with the buffer in ordinary memory), and another negative value for an
// invalid descriptor.
TCRT_API int tcrt_launch(const tcrt_launch_desc *desc);

#define TCRT_ERROR_NO_TCM (-2)

//===----------------------------------------------------------------------===//
// Kernel ABI
//===----------------------------------------------------------------------===//

// Body of a parallel region: called once per logical thread id `tid` in
// [0, nthreads).
typedef void (*cpar_region_fn)(void *ctx, int64_t tid, int64_t nthreads);

// Runs fn(ctx, tid, nthreads) for every tid in [0, nthreads) and returns when
// all calls have finished (fork/join).  Called from compiled kernels, see the
// `cpar` dialect in buddy-mlir.
//
// Inside a program instance of a launch with warps > 1, the calls are spread
// over the CPUs of the instance's slot: the calling thread runs tid 0 and the
// other threads of the slot run the remaining ids round robin.  Everywhere else
// (single CPU slot, nested regions, calls outside of tcrt_launch) the calls are
// made one after the other on the calling thread.
TCRT_API void __cpar_parallel(cpar_region_fn fn, void *ctx, int64_t nthreads);

// Queries kernels can make about the thread executing them
// (triton.language.extra.cpu.current_cpu / os_thread_id).
// CPU the calling thread is running on (sched_getcpu), -1 if unknown.
TCRT_API int32_t triton_cpu_current_cpu(void);
// Linux thread id (gettid) of the calling thread, 0 if unknown.
TCRT_API int32_t triton_cpu_os_thread_id(void);

// Base of the triton.language.extra.cpu.alloc buffers of the calling thread
// (`bytes` in total, at static offsets computed by buddy-mlir
// --cpu-tcm-assign), 128-byte aligned.  On the AI cores of the SpacemiT K3
// this is the TCM block next to the core of the calling worker, acquired by
// the worker on first use and released at exit; otherwise, or if the block is
// not available, ordinary memory of the calling thread.  The memory is reused
// by every program instance (and parallel region share) the thread executes.
// TRITON_CPU_TCM=0 always uses ordinary memory, TRITON_CPU_TCM=require makes
// tcrt_launch() report TCRT_ERROR_NO_TCM when the TCM was not used.
TCRT_API void *triton_cpu_tcm_base(int64_t bytes);
// Same, for buffers that the threads of a cpu.parallel loop share: also warns
// once if the program instance runs on cores that are not next to the TCM
// block of the calling thread (use num_warps=2 on the K3).
TCRT_API void *triton_cpu_tcm_base_shared(int64_t bytes);

//===----------------------------------------------------------------------===//
// Introspection (tests and debugging)
//===----------------------------------------------------------------------===//

TCRT_API int tcrt_num_cpus(void);
TCRT_API int tcrt_num_domains(void);
// Writes up to `cap` CPU ids of `domain` into `cpus`; returns the domain size
// or -1 for an invalid domain.
TCRT_API int tcrt_domain_cpus(int domain, int *cpus, int cap);
// Number of slots, i.e. the maximum number of concurrent program instances, of
// a launch with `warps` threads per program instance.
TCRT_API int tcrt_num_slots(int warps);
// 1 if the pool runs on the AI cores (the CPUs of tcrt_domain_cpus are then AI
// cores and kernel code never runs on the calling thread), 0 otherwise.
TCRT_API int tcrt_ai_cores(void);
// 1 if the workers put cpu.alloc buffers in the TCM (AI cores, libspine_tcm
// available, TRITON_CPU_TCM not "0"), 0 otherwise.
TCRT_API int tcrt_tcm_available(void);
// Makes every worker release its TCM block (e.g. for another process); the
// next kernel using cpu.alloc acquires them again.  Must not be called during
// a launch.
TCRT_API void tcrt_tcm_release(void);
// Size of one TCM block in bytes, 0 without TCM.
TCRT_API int64_t tcrt_tcm_block_bytes(void);
// TCM state of the worker pinned to `cpu`: -1 if there is no such worker or it
// has not used cpu.alloc yet, 1 if it owns TCM block *block (mapped at *base),
// 0 if its buffers are in ordinary memory.
TCRT_API int tcrt_worker_tcm(int cpu, int *block, void **base);

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRITON_CPU_RT_H
