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
// Returns 0 on success and a negative value for an invalid descriptor.
TCRT_API int tcrt_launch(const tcrt_launch_desc *desc);

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

#ifdef __cplusplus
} // extern "C"
#endif

#endif // TRITON_CPU_RT_H
