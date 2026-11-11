//===- triton_cpu_rt.cpp - triton-riscv CPU runtime -------------*- C++ -*-===//
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
// Worker pool that executes Triton launch grids on the CPU. See
// triton_cpu_rt.h for the ABI and docs/09-Program-Instance-Parallelism.md
// and docs/10-Program-Instance-Threads.md for the design.
//
// Structure:
//   * Configuration: the compute CPUs of the board configuration, grouped
//     into domains (CPUs sharing a cache: the board configuration's L2
//     groups, or what TRITON_CPAR_CPUS lists), given by tcrt_configure().
//     Nothing about the board is detected here; backend/board.py does it.
//   * AI cores (SpacemiT K3): the A100 cores are outside the affinity of
//     ordinary threads; a thread gets access to them, and loses access to the
//     other cores, by writing its id to /proc/set_ai_thread. With
//     enter_ai_cores a short lived thread does that and creates the workers,
//     which inherit the access.
//   * Pool: one worker thread per compute CPU, pinned to that CPU at creation
//     and never migrated afterwards. Workers sleep on a per-worker mailbox.
//     Kernel code only runs on workers, never on the calling thread, so it
//     never runs on a CPU outside of the configuration.
//   * Scratchpad: the first worker acquires the board's whole scratchpad
//     when the pool starts and releases it at exit (the driver requires the
//     same thread for both). Every slot gets the share of its CPUs.
//   * Slots: for a launch with W threads per program instance the CPUs are
//     cut into slots of W consecutive workers (see tcrt_launch). The first
//     worker of a slot is its leader and executes program instances, the
//     others are helpers that only run parallel regions (__cpar_parallel)
//     of the leader's instance.
//   * Launch: the calling thread posts the grid to the leaders of P slots,
//     which grab program instances from a shared atomic counter (dynamic
//     schedule with a chunk of one instance) and count down a latch when the
//     grid is drained. The calling thread only waits on the latch.
//
// Waiting (workers on their mailbox, the caller and leaders on latches) spins
// for a bounded number of iterations and then sleeps on a futex.
//
//===----------------------------------------------------------------------===//

#include "triton_cpu_rt.h"

#include TCRT_SPM_DRIVER_H

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#include <fcntl.h>
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>

namespace {

// Iterations a waiting thread spins before it sleeps.
constexpr uint32_t kSpins = 4000;

//===----------------------------------------------------------------------===//
// Small platform helpers
//===----------------------------------------------------------------------===//

int64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int currentCpu() { return sched_getcpu(); }

int64_t osThreadId() { return static_cast<int64_t>(syscall(SYS_gettid)); }

inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__riscv)
  // Zihintpause `pause` (fence w, 0). Cores without the extension execute it
  // as a plain hint, so it is safe everywhere.
  asm volatile(".4byte 0x0100000f" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

void warnOnce(std::atomic<bool> &flag, const char *message) {
  if (!flag.exchange(true))
    std::fprintf(stderr, "triton-cpu-rt: warning: %s\n", message);
}

//===----------------------------------------------------------------------===//
// Event / Latch
//===----------------------------------------------------------------------===//

static_assert(sizeof(std::atomic<uint32_t>) == sizeof(uint32_t),
              "futex word must be a plain 32-bit integer");

// Monotonic sequence number that threads can wait on for a change.
class Event {
public:
  uint32_t epoch() const { return seq_.load(std::memory_order_acquire); }

  // Publishes everything written before the call to threads that observe the
  // new epoch, and wakes sleepers.
  void notify() {
    seq_.fetch_add(1, std::memory_order_seq_cst);
    // Pairs with the seq_cst increment of waiters_ in wait(): either the
    // waiter sees the new epoch before sleeping, or we see the waiter here.
    if (waiters_.load(std::memory_order_seq_cst) != 0)
      syscall(SYS_futex, reinterpret_cast<uint32_t *>(&seq_),
              FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
  }

  // Blocks until epoch() != seen; spins first.
  void wait(uint32_t seen) {
    for (uint32_t i = 0; i < kSpins; ++i) {
      if (seq_.load(std::memory_order_acquire) != seen)
        return;
      cpuRelax();
    }
    waiters_.fetch_add(1, std::memory_order_seq_cst);
    while (seq_.load(std::memory_order_seq_cst) == seen)
      syscall(SYS_futex, reinterpret_cast<uint32_t *>(&seq_),
              FUTEX_WAIT_PRIVATE, seen, nullptr, nullptr, 0);
    waiters_.fetch_sub(1, std::memory_order_relaxed);
  }

private:
  std::atomic<uint32_t> seq_{0};
  std::atomic<uint32_t> waiters_{0};
};

// Count-down latch. wait() returns once arrive() was called `count` times;
// everything written by the arriving threads is visible to the waiter.
//
// A latch is reused from one round to the next, and the last arriving thread
// may still be inside done_.notify() when the waiter of the round already
// returned. Latches therefore live as long as the pool, never on a stack
// frame of a launch. A late notify() only causes a spurious wake-up of the
// next round's waiter, which re-checks the count.
class Latch {
public:
  void reset(int count) { count_.store(count, std::memory_order_relaxed); }

  void arrive() {
    if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1)
      done_.notify();
  }

  void wait() {
    for (;;) {
      const uint32_t seen = done_.epoch();
      if (count_.load(std::memory_order_acquire) == 0)
        return;
      done_.wait(seen);
    }
  }

private:
  std::atomic<int> count_{0};
  Event done_;
};

//===----------------------------------------------------------------------===//
// AI cores (SpacemiT K3)
//===----------------------------------------------------------------------===//

const char *const kAiThreadPath = "/proc/set_ai_thread";

// Moves the calling thread to the AI cores. The move is one way: afterwards
// the thread (and every thread it creates) can only run on AI cores. The
// migration happens inside the write() system call, where the vector state is
// discarded anyway, so it is safe even across cores with different VLEN.
bool enableAiCoresForThisThread() {
  const int fd = open(kAiThreadPath, O_WRONLY | O_CLOEXEC);
  if (fd < 0)
    return false;
  char text[32];
  const int length = std::snprintf(text, sizeof(text), "%lld",
                                   static_cast<long long>(osThreadId()));
  const ssize_t written = write(fd, text, static_cast<size_t>(length));
  close(fd);
  return written == length;
}

struct AiThreadTask {
  const std::function<void()> *body = nullptr;
  bool enabled = false;
};

void *aiThreadMain(void *raw) {
  AiThreadTask *task = static_cast<AiThreadTask *>(raw);
  task->enabled = enableAiCoresForThisThread();
  if (task->enabled)
    (*task->body)();
  return nullptr;
}

// Runs `body` on a new, short lived thread that has been moved to the AI
// cores and waits for it. The calling thread is left untouched. Returns
// false (and does not run `body`) if the thread could not be created or
// moved.
bool runOnAiThread(const std::function<void()> &body) {
  AiThreadTask task;
  task.body = &body;
  // The thread and the workers it creates must not take asynchronous signals
  // away from the application threads.
  sigset_t all, previous;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &previous);
  pthread_t thread;
  const int rc = pthread_create(&thread, nullptr, aiThreadMain, &task);
  pthread_sigmask(SIG_SETMASK, &previous, nullptr);
  if (rc != 0)
    return false;
  pthread_join(thread, nullptr);
  return task.enabled;
}

//===----------------------------------------------------------------------===//
// Worker pool data structures
//===----------------------------------------------------------------------===//

struct TraceEntry {
  int64_t startNs;
  int64_t endNs;
  int threadId;
  int cpu;
  int64_t osTid;
};

// One call fn(ctx, tid, n) of a parallel region.
struct RegionTraceEntry {
  int64_t instance;
  int region;
  int64_t tid;
  int64_t nthreads;
  int64_t osTid;
  int cpu;
  int64_t startNs;
  int64_t endNs;
};

struct Slot;

struct GridJob {
  tcrt_instance_fn fn = nullptr;
  void *args = nullptr;
  int gridY = 1;
  int gridZ = 1;
  int64_t total = 0;
  std::atomic<int64_t> next{0};
  // Owned by the pool, not by the job (see Latch). The job itself lives on
  // the stack of tcrt_launch().
  Latch *done = nullptr;
  TraceEntry *trace = nullptr;
  // Region records of instances executed inline by a nested launch.
  std::vector<RegionTraceEntry> inlineRegionTrace;
};

// A parallel region posted by a slot leader to its helpers. Lives on the
// leader's stack for the duration of __cpar_parallel().
struct RegionTask {
  cpar_region_fn fn = nullptr;
  void *ctx = nullptr;
  int64_t nthreads = 0;
  int64_t stride = 1; // number of threads sharing the logical ids
  Latch *done = nullptr;
  bool tracing = false;
  int64_t instance = -1;
  int region = -1;
};

enum class MessageKind { Grid, Region, Call };

struct alignas(64) Worker {
  int cpu = -1;
  Event mail;
  // Message, written by the sender before mail.notify() and read by the
  // worker after it observed the new epoch.
  MessageKind kind = MessageKind::Grid;
  GridJob *job = nullptr;                      // Grid
  int rank = 0;                                // Grid: rank of the slot
  Slot *slot = nullptr;                        // Grid: slot led by this worker
  RegionTask *task = nullptr;                  // Region
  int member = 0;                              // Region: index in the slot
  const std::function<void()> *call = nullptr; // Call
  Latch *callDone = nullptr;                   // Call
  // Region records of this worker during the current launch. Cleared by the
  // dispatcher before the launch, read by it after the launch.
  std::vector<RegionTraceEntry> regionTrace;
};

// Consecutive CPUs that run one program instance at a time.
struct Slot {
  std::vector<Worker *> members; // members[0] is the leader
  Latch regionDone;
  // Share of the slot's CPUs in the scratchpad: 0 bytes if they have none,
  // `spm` null if the scratchpad is not held.
  int64_t spmOffset = 0;
  int64_t spmBytes = 0;
  char *spm = nullptr;
};

struct Pool {
  // Configuration.
  std::vector<int> cpus;
  std::vector<int> domainSizes;
  bool ai = false;
  int64_t spmBytesPerCpu = 0;
  int spmFirstCpu = 0;
  // Size of the scratchpad the shares are cut from: the driver's region
  // when held, else the sum of the shares of the configured CPUs from
  // spmFirstCpu on (so that shares are known even when the hold failed).
  int64_t spmTotalBytes = 0;
  // State.
  std::vector<Worker *> workers; // one per started CPU, in config order
  std::vector<int> workerDomain; // domain index of each worker
  Latch gridDone;
  Latch callDone;
  spm_hold_t spm{};
  int spmCode = SPM_ERR_NO_DRIVER;
  Worker *spmHolder = nullptr;
  // Slot layouts per number of warps, in launch order (round robin over the
  // domains, so that a launch with fewer instances than slots spreads over
  // the domains). Built on first use, kept for the lifetime of the pool.
  std::map<int, std::vector<Slot *>> layouts;
};

thread_local Worker *tlsWorker = nullptr;
// Slot whose leader is the current thread, while it executes instances.
thread_local Slot *tlsSlot = nullptr;
// True while the current thread executes a parallel region.
thread_local bool tlsInRegion = false;
// Tracing state of the instance executed by the current thread.
thread_local bool tlsTracing = false;
thread_local int64_t tlsInstance = -1;
thread_local int tlsRegionCount = 0;
thread_local std::vector<RegionTraceEntry> *tlsRegionSink = nullptr;

//===----------------------------------------------------------------------===//
// Execution
//===----------------------------------------------------------------------===//

// Runs the logical ids member, member + stride, ... of `task`.
void runRegionShare(const RegionTask &task, int64_t member) {
  for (int64_t tid = member; tid < task.nthreads; tid += task.stride) {
    if (!task.tracing || tlsRegionSink == nullptr) {
      task.fn(task.ctx, tid, task.nthreads);
      continue;
    }
    RegionTraceEntry entry;
    entry.instance = task.instance;
    entry.region = task.region;
    entry.tid = tid;
    entry.nthreads = task.nthreads;
    entry.osTid = osThreadId();
    entry.cpu = currentCpu();
    entry.startNs = nowNs();
    task.fn(task.ctx, tid, task.nthreads);
    entry.endNs = nowNs();
    tlsRegionSink->push_back(entry);
  }
}

// Runs instances of `job` until the grid is drained.
void runInstances(GridJob &job, int rank, void *spm) {
  const int64_t gridYZ = static_cast<int64_t>(job.gridY) * job.gridZ;
  for (;;) {
    const int64_t id = job.next.fetch_add(1, std::memory_order_relaxed);
    if (id >= job.total)
      return;
    // Same flattening as a serial loop nest: x outer, z inner.
    const int x = static_cast<int>(id / gridYZ);
    const int64_t yz = id % gridYZ;
    const int y = static_cast<int>(yz / job.gridZ);
    const int z = static_cast<int>(yz % job.gridZ);
    tlsInstance = id;
    tlsRegionCount = 0;
    TraceEntry *entry = job.trace != nullptr ? &job.trace[id] : nullptr;
    if (entry != nullptr) {
      entry->startNs = nowNs();
      entry->threadId = rank;
      entry->cpu = currentCpu();
      entry->osTid = osThreadId();
    }
    job.fn(job.args, spm, x, y, z);
    if (entry != nullptr)
      entry->endNs = nowNs();
  }
}

// Thread local state of a thread that executes instances of a launch.
struct InstanceScope {
  InstanceScope(Slot *slot, bool tracing, std::vector<RegionTraceEntry> *sink)
      : savedSlot(tlsSlot), savedTracing(tlsTracing),
        savedInstance(tlsInstance), savedCount(tlsRegionCount),
        savedSink(tlsRegionSink), savedInRegion(tlsInRegion) {
    tlsSlot = slot;
    tlsTracing = tracing;
    tlsRegionSink = sink;
    tlsInRegion = false;
  }
  ~InstanceScope() {
    tlsSlot = savedSlot;
    tlsTracing = savedTracing;
    tlsInstance = savedInstance;
    tlsRegionCount = savedCount;
    tlsRegionSink = savedSink;
    tlsInRegion = savedInRegion;
  }
  InstanceScope(const InstanceScope &) = delete;
  InstanceScope &operator=(const InstanceScope &) = delete;

  Slot *savedSlot;
  bool savedTracing;
  int64_t savedInstance;
  int savedCount;
  std::vector<RegionTraceEntry> *savedSink;
  bool savedInRegion;
};

void *workerMain(void *raw) {
  Worker *self = static_cast<Worker *>(raw);
  tlsWorker = self;
  uint32_t seen = 0; // the mailbox epoch at creation time
  for (;;) {
    self->mail.wait(seen);
    seen = self->mail.epoch();
    switch (self->kind) {
    case MessageKind::Grid: {
      GridJob *job = self->job;
      Latch *done = job->done;
      {
        InstanceScope scope(self->slot, job->trace != nullptr,
                            &self->regionTrace);
        runInstances(*job, self->rank, self->slot->spm);
      }
      // `job` may be gone as soon as the latch opens; do not touch it after.
      done->arrive();
      break;
    }
    case MessageKind::Region: {
      RegionTask *task = self->task;
      Latch *done = task->done;
      tlsInRegion = true;
      tlsRegionSink = &self->regionTrace;
      runRegionShare(*task, self->member);
      tlsInRegion = false;
      // `task` lives on the leader's stack; do not touch it after.
      done->arrive();
      break;
    }
    case MessageKind::Call: {
      Latch *done = self->callDone;
      (*self->call)();
      done->arrive();
      break;
    }
    }
  }
  return nullptr;
}

std::atomic<bool> warnedAi{false};
std::atomic<bool> warnedCross{false};

// Starts a detached worker pinned to `worker->cpu`. Asynchronous signals are
// blocked in the worker so that they keep being delivered to the threads of
// the embedding application (e.g. the Python main thread). A worker that
// cannot be pinned is not started: unpinned it could run kernel code on a
// CPU outside of the configuration.
bool startWorker(Worker *worker) {
  sigset_t all, previous;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &previous);

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(worker->cpu, &set);
  int rc = pthread_attr_setaffinity_np(&attr, sizeof(set), &set);
  pthread_t thread;
  if (rc == 0)
    rc = pthread_create(&thread, &attr, workerMain, worker);
  pthread_attr_destroy(&attr);
  pthread_sigmask(SIG_SETMASK, &previous, nullptr);
  return rc == 0;
}

void createWorkers(Pool &pool) {
  size_t index = 0;
  for (size_t d = 0; d < pool.domainSizes.size(); ++d) {
    for (int i = 0; i < pool.domainSizes[d]; ++i, ++index) {
      Worker *worker = new Worker();
      worker->cpu = pool.cpus[index];
      if (!startWorker(worker)) {
        delete worker;
        std::fprintf(stderr,
                     "triton-cpu-rt: warning: no worker thread on CPU %d "
                     "(cannot pin a thread to it); the CPU is left out\n",
                     pool.cpus[index]);
        continue;
      }
      pool.workers.push_back(worker);
      pool.workerDomain.push_back(static_cast<int>(d));
    }
  }
}

// Runs `fn` on `worker` and waits for it. Requires gLock.
void callOn(Pool &pool, Worker *worker, const std::function<void()> &fn) {
  pool.callDone.reset(1);
  worker->kind = MessageKind::Call;
  worker->call = &fn;
  worker->callDone = &pool.callDone;
  worker->mail.notify();
  pool.callDone.wait();
}

// Slots of a launch with `warps` threads per instance, in launch order.
const std::vector<Slot *> &getLayout(Pool &pool, int warps) {
  auto found = pool.layouts.find(warps);
  if (found != pool.layouts.end())
    return found->second;

  const size_t n = pool.workers.size();
  const size_t size = std::min<size_t>(std::max(warps, 1), n);
  // End (exclusive) of the domain of each worker.
  std::vector<size_t> domainEnd(n);
  for (size_t i = n; i-- > 0;)
    domainEnd[i] =
        (i + 1 < n && pool.workerDomain[i + 1] == pool.workerDomain[i])
            ? domainEnd[i + 1]
            : i + 1;

  std::vector<std::vector<Slot *>> byStartDomain(pool.domainSizes.size());
  bool crosses = false;
  size_t pos = 0;
  while (size > 0 && pos < n) {
    const bool domainStart =
        pos == 0 || pool.workerDomain[pos - 1] != pool.workerDomain[pos];
    // Not enough room left in a domain already in use: the next slot starts
    // at the next domain and the rest stays idle.
    if (!domainStart && domainEnd[pos] - pos < size) {
      pos = domainEnd[pos];
      continue;
    }
    if (n - pos < size)
      break;
    Slot *slot = new Slot();
    slot->members.assign(pool.workers.begin() + pos,
                         pool.workers.begin() + pos + size);
    crosses |= pool.workerDomain[pos] != pool.workerDomain[pos + size - 1];
    // The slot's share of the scratchpad: the shares of its CPUs, which must
    // be consecutive CPUs that have one.
    if (pool.spmBytesPerCpu > 0) {
      const int first = slot->members.front()->cpu - pool.spmFirstCpu;
      bool contiguous = first >= 0;
      for (size_t m = 0; m < size; ++m)
        contiguous &= slot->members[m]->cpu ==
                      slot->members.front()->cpu + static_cast<int>(m);
      const int64_t end =
          (static_cast<int64_t>(first) + static_cast<int64_t>(size)) *
          pool.spmBytesPerCpu;
      if (contiguous && end <= pool.spmTotalBytes) {
        slot->spmOffset = static_cast<int64_t>(first) * pool.spmBytesPerCpu;
        slot->spmBytes = static_cast<int64_t>(size) * pool.spmBytesPerCpu;
        if (pool.spmCode == SPM_OK)
          slot->spm = static_cast<char *>(pool.spm.base) + slot->spmOffset;
      }
    }
    byStartDomain[pool.workerDomain[pos]].push_back(slot);
    pos += size;
  }
  if (crosses)
    warnOnce(warnedCross,
             "the threads of a program instance span several domains of the "
             "CPU configuration (num_warps is larger than a domain)");

  std::vector<Slot *> layout;
  for (size_t i = 0;; ++i) {
    bool any = false;
    for (const std::vector<Slot *> &slots : byStartDomain)
      if (i < slots.size()) {
        layout.push_back(slots[i]);
        any = true;
      }
    if (!any)
      break;
  }
  return pool.layouts.emplace(warps, std::move(layout)).first->second;
}

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Serializes configuration and launches. Launchers hold the GIL for the
// whole launch; the lock covers callers that do not.
pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;

struct GlobalGuard {
  GlobalGuard() { pthread_mutex_lock(&gLock); }
  ~GlobalGuard() { pthread_mutex_unlock(&gLock); }
  GlobalGuard(const GlobalGuard &) = delete;
  GlobalGuard &operator=(const GlobalGuard &) = delete;
};

Pool *gPool = nullptr;
// Details of the last launch that failed for lack of scratchpad.
tcrt_spm_status_t gLastSpmFailure{};

// The scratchpad is released by the thread that acquired it, at exit.
void releaseSpmAtExit() {
  GlobalGuard guard;
  Pool *pool = gPool;
  if (pool == nullptr || pool->spmHolder == nullptr || pool->spmCode != SPM_OK)
    return;
  std::function<void()> release = [pool] { spm_release(&pool->spm); };
  callOn(*pool, pool->spmHolder, release);
  pool->spmCode = SPM_ERR_NO_DRIVER;
}

void startPool(Pool &pool) {
  if (!pool.ai) {
    createWorkers(pool);
  } else if (!runOnAiThread([&pool] { createWorkers(pool); })) {
    // Workers created by a thread that has access to the AI cores inherit
    // the access, so they are pinned to their AI core from the start and
    // never execute anything on another core. The creating thread is joined
    // before runOnAiThread() returns, which publishes its writes to `pool`.
    warnOnce(warnedAi, "could not move the worker threads to the AI cores "
                       "(/proc/set_ai_thread)");
  }
  if (pool.spmBytesPerCpu > 0 && !pool.workers.empty()) {
    pool.spmHolder = pool.workers.front();
    std::function<void()> hold = [&pool] {
      pool.spmCode = spm_hold(&pool.spm);
    };
    callOn(pool, pool.spmHolder, hold);
    if (pool.spmCode == SPM_OK)
      std::atexit(releaseSpmAtExit);
  }
  // The driver reports its size even when the hold failed.
  pool.spmTotalBytes = pool.spm.bytes;
}

bool sameConfig(const Pool &pool, const tcrt_config &config) {
  return pool.cpus ==
             std::vector<int>(config.cpus, config.cpus + config.ncpus) &&
         pool.domainSizes ==
             std::vector<int>(config.domain_sizes,
                              config.domain_sizes + config.ndomains) &&
         pool.ai == (config.enter_ai_cores != 0) &&
         pool.spmBytesPerCpu == config.spm_bytes_per_cpu &&
         pool.spmFirstCpu == config.spm_first_cpu;
}

void writeTrace(const char *path, const std::vector<TraceEntry> &entries) {
  FILE *file = std::fopen(path, "w");
  if (file == nullptr)
    return;
  std::fputs("instance,start_ns,end_ns,thread_id,cpu,os_tid\n", file);
  for (size_t i = 0; i < entries.size(); ++i)
    std::fprintf(file, "%zu,%lld,%lld,%d,%d,%lld\n", i,
                 static_cast<long long>(entries[i].startNs),
                 static_cast<long long>(entries[i].endNs), entries[i].threadId,
                 entries[i].cpu, static_cast<long long>(entries[i].osTid));
  std::fclose(file);
}

void writeRegionTrace(const char *tracePath,
                      std::vector<RegionTraceEntry> &entries) {
  std::sort(entries.begin(), entries.end(),
            [](const RegionTraceEntry &a, const RegionTraceEntry &b) {
              return std::tie(a.instance, a.region, a.tid) <
                     std::tie(b.instance, b.region, b.tid);
            });
  const std::string path = std::string(tracePath) + ".threads";
  FILE *file = std::fopen(path.c_str(), "w");
  if (file == nullptr)
    return;
  std::fputs("instance,region,tid,nthreads,os_tid,cpu,start_ns,end_ns\n", file);
  for (const RegionTraceEntry &e : entries)
    std::fprintf(
        file, "%lld,%d,%lld,%lld,%lld,%d,%lld,%lld\n",
        static_cast<long long>(e.instance), e.region,
        static_cast<long long>(e.tid), static_cast<long long>(e.nthreads),
        static_cast<long long>(e.osTid), e.cpu,
        static_cast<long long>(e.startNs), static_cast<long long>(e.endNs));
  std::fclose(file);
}

// Checks that every slot of `layout` that the launch uses can provide the
// kernel's scratchpad. Fills gLastSpmFailure and returns false otherwise.
bool checkSpm(const Pool &pool, const std::vector<Slot *> &layout,
              int64_t slots, int64_t bytes, int64_t align) {
  tcrt_spm_status_t &failure = gLastSpmFailure;
  failure = tcrt_spm_status_t{};
  failure.busy_block = pool.spm.busy_block;
  failure.sys_errno = pool.spm.sys_errno;
  failure.bytes = pool.spm.bytes;
  failure.slot_cpu = -1;
  if (pool.spmBytesPerCpu == 0) {
    failure.code = SPM_ERR_NO_DRIVER;
    return false;
  }
  // What the configuration cannot provide comes first; whether the
  // scratchpad is held last.
  for (int64_t i = 0; i < slots; ++i) {
    const Slot *slot = layout[i];
    failure.slot_cpu = slot->members.front()->cpu;
    failure.slot_bytes = slot->spmBytes;
    if (slot->spmBytes == 0) {
      failure.code = SPM_ERR_CPU;
      return false;
    }
    if (slot->spmBytes < bytes) {
      failure.code = SPM_ERR_CAPACITY;
      return false;
    }
  }
  if (align > sysconf(_SC_PAGESIZE)) {
    failure.code = SPM_ERR_ALIGN;
    return false;
  }
  if (pool.spmCode != SPM_OK) {
    failure.code = pool.spmCode;
    return false;
  }
  failure.code = SPM_OK;
  return true;
}

} // namespace

//===----------------------------------------------------------------------===//
// ABI
//===----------------------------------------------------------------------===//

extern "C" TCRT_API int tcrt_configure(const tcrt_config *config) {
  if (config == nullptr || config->abi_version != TCRT_ABI_VERSION ||
      config->ncpus <= 0 || config->cpus == nullptr || config->ndomains <= 0 ||
      config->domain_sizes == nullptr)
    return TCRT_ERROR_INVALID;
  int total = 0;
  for (int d = 0; d < config->ndomains; ++d) {
    if (config->domain_sizes[d] <= 0)
      return TCRT_ERROR_INVALID;
    total += config->domain_sizes[d];
  }
  if (total != config->ncpus)
    return TCRT_ERROR_INVALID;

  GlobalGuard guard;
  if (gPool != nullptr)
    return sameConfig(*gPool, *config) ? TCRT_OK : TCRT_ERROR_RECONFIGURE;
  Pool *pool = new Pool();
  pool->cpus.assign(config->cpus, config->cpus + config->ncpus);
  pool->domainSizes.assign(config->domain_sizes,
                           config->domain_sizes + config->ndomains);
  pool->ai = config->enter_ai_cores != 0;
  pool->spmBytesPerCpu = config->spm_bytes_per_cpu;
  pool->spmFirstCpu = config->spm_first_cpu;
  startPool(*pool);
  gPool = pool;
  return TCRT_OK;
}

extern "C" TCRT_API void tcrt_spm_status(tcrt_spm_status_t *status) {
  if (status == nullptr)
    return;
  GlobalGuard guard;
  *status = gLastSpmFailure;
  if (gPool != nullptr && gLastSpmFailure.code == SPM_OK) {
    status->code = gPool->spmCode;
    status->busy_block = gPool->spm.busy_block;
    status->sys_errno = gPool->spm.sys_errno;
    status->bytes = gPool->spm.bytes;
  }
}

extern "C" TCRT_API int tcrt_launch(const tcrt_launch_desc *desc) {
  if (desc == nullptr || desc->abi_version != TCRT_ABI_VERSION ||
      desc->fn == nullptr)
    return TCRT_ERROR_INVALID;
  if (desc->grid[0] <= 0 || desc->grid[1] <= 0 || desc->grid[2] <= 0)
    return TCRT_OK;
  const int warps = std::max(desc->warps, 1);

  GridJob job;
  job.fn = desc->fn;
  job.args = desc->args;
  job.gridY = desc->grid[1];
  job.gridZ = desc->grid[2];
  job.total =
      static_cast<int64_t>(desc->grid[0]) * desc->grid[1] * desc->grid[2];

  const bool tracing = desc->trace_path != nullptr && desc->trace_path[0];
  std::vector<TraceEntry> trace(tracing ? static_cast<size_t>(job.total) : 0);
  job.trace = tracing ? trace.data() : nullptr;

  std::vector<RegionTraceEntry> regionTrace;
  if (tlsWorker != nullptr) {
    // A launch issued from a pool worker (nested launch) runs inline: the
    // pool is busy with the enclosing launch. It uses the scratchpad of the
    // enclosing slot.
    Slot *slot = tlsSlot;
    if (desc->spm_bytes > 0 && (slot == nullptr || slot->spm == nullptr ||
                                slot->spmBytes < desc->spm_bytes)) {
      GlobalGuard guard;
      gLastSpmFailure = tcrt_spm_status_t{};
      gLastSpmFailure.code = slot == nullptr || slot->spm == nullptr
                                 ? SPM_ERR_CPU
                                 : SPM_ERR_CAPACITY;
      gLastSpmFailure.slot_cpu = tlsWorker->cpu;
      gLastSpmFailure.slot_bytes = slot ? slot->spmBytes : 0;
      return TCRT_ERROR_SPM;
    }
    InstanceScope scope(nullptr, tracing, &job.inlineRegionTrace);
    runInstances(job, 0, slot ? slot->spm : nullptr);
  } else {
    GlobalGuard guard;
    if (gPool == nullptr || gPool->workers.empty())
      return TCRT_ERROR_INVALID;
    Pool &pool = *gPool;
    const std::vector<Slot *> &layout = getLayout(pool, warps);
    const int64_t parallel =
        std::min<int64_t>(job.total, static_cast<int64_t>(layout.size()));
    if (parallel == 0)
      return TCRT_ERROR_INVALID;
    if (desc->spm_bytes > 0 &&
        !checkSpm(pool, layout, parallel, desc->spm_bytes, desc->spm_align))
      return TCRT_ERROR_SPM;

    const int leaders = static_cast<int>(parallel);
    job.done = &pool.gridDone;
    job.done->reset(leaders);
    if (tracing)
      for (int rank = 0; rank < leaders; ++rank)
        for (Worker *member : layout[rank]->members)
          member->regionTrace.clear();
    for (int rank = 0; rank < leaders; ++rank) {
      Slot *slot = layout[rank];
      Worker *leader = slot->members[0];
      leader->kind = MessageKind::Grid;
      leader->job = &job;
      leader->rank = rank;
      leader->slot = slot;
      leader->mail.notify();
    }
    job.done->wait();
    if (tracing)
      for (int rank = 0; rank < leaders; ++rank)
        for (Worker *member : layout[rank]->members)
          regionTrace.insert(regionTrace.end(), member->regionTrace.begin(),
                             member->regionTrace.end());
  }

  if (tracing) {
    writeTrace(desc->trace_path, trace);
    regionTrace.insert(regionTrace.end(), job.inlineRegionTrace.begin(),
                       job.inlineRegionTrace.end());
    writeRegionTrace(desc->trace_path, regionTrace);
  }
  return TCRT_OK;
}

extern "C" TCRT_API void __cpar_parallel(cpar_region_fn fn, void *ctx,
                                         int64_t nthreads) {
  if (fn == nullptr || nthreads <= 0)
    return;

  RegionTask task;
  task.fn = fn;
  task.ctx = ctx;
  task.nthreads = nthreads;
  task.tracing = tlsTracing;
  task.instance = tlsInstance;
  task.region = tlsRegionCount++;

  // Only the leader of a multi-CPU slot forks; nested regions and calls from
  // any other thread run serially on the calling thread.
  Slot *slot = tlsInRegion ? nullptr : tlsSlot;
  const int64_t threads =
      slot == nullptr
          ? 1
          : std::min<int64_t>(nthreads,
                              static_cast<int64_t>(slot->members.size()));
  task.stride = threads;

  const bool savedInRegion = tlsInRegion;
  if (threads <= 1) {
    tlsInRegion = true;
    runRegionShare(task, 0);
    tlsInRegion = savedInRegion;
    return;
  }

  task.done = &slot->regionDone;
  task.done->reset(static_cast<int>(threads - 1));
  for (int64_t member = 1; member < threads; ++member) {
    Worker *helper = slot->members[member];
    helper->kind = MessageKind::Region;
    helper->task = &task;
    helper->member = static_cast<int>(member);
    helper->mail.notify();
  }
  tlsInRegion = true;
  runRegionShare(task, 0);
  tlsInRegion = savedInRegion;
  task.done->wait();
}

extern "C" TCRT_API int32_t tcrt_current_cpu(void) {
  return static_cast<int32_t>(currentCpu());
}

extern "C" TCRT_API int32_t tcrt_os_thread_id(void) {
  return static_cast<int32_t>(osThreadId());
}

extern "C" TCRT_API int tcrt_num_slots(int warps) {
  GlobalGuard guard;
  if (gPool == nullptr)
    return 0;
  return static_cast<int>(getLayout(*gPool, std::max(warps, 1)).size());
}

extern "C" TCRT_API int tcrt_slot_cpus(int warps, int slot, int *cpus,
                                       int cap) {
  GlobalGuard guard;
  if (gPool == nullptr)
    return -1;
  const std::vector<Slot *> &layout = getLayout(*gPool, std::max(warps, 1));
  if (slot < 0 || static_cast<size_t>(slot) >= layout.size())
    return -1;
  const std::vector<Worker *> &members = layout[slot]->members;
  for (int i = 0; i < cap && static_cast<size_t>(i) < members.size(); ++i)
    cpus[i] = members[i]->cpu;
  return static_cast<int>(members.size());
}

extern "C" TCRT_API int tcrt_num_workers(void) {
  GlobalGuard guard;
  return gPool == nullptr ? 0 : static_cast<int>(gPool->workers.size());
}
