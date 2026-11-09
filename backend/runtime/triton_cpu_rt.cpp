//===- triton_cpu_rt.cpp - triton-riscv CPU runtime -------------*- C++ -*-===//
//
// Worker pool that executes Triton launch grids on the CPU.  See
// triton_cpu_rt.h for the ABI, docs/05-Program-Instance-Parallelism.md and
// docs/06-Program-Instance-Threads.md for the design.
//
// Structure:
//   * Topology: the CPUs available to the process (sched_getaffinity) grouped
//     into "domains", i.e. CPUs sharing an L2 cache (one cluster on the K1).
//   * Pool: one worker thread per available CPU, pinned to that CPU at creation
//     and never migrated afterwards.  Workers sleep on a per-worker mailbox.
//   * Slots: for a launch with W threads per program instance every domain is
//     cut into slots of min(W, domain size) workers.  The first worker of a
//     slot is its leader and executes program instances, the others are
//     helpers that only run parallel regions (__cpar_parallel) of the
//     leader's instance.
//   * Launch: the calling thread posts the grid to the leaders of P slots,
//     which grab program instances from a shared atomic counter (dynamic
//     schedule with a chunk of one instance) and count down a latch when the
//     grid is drained.  The calling thread itself is never pinned; it only
//     waits on the latch.
//
// Waiting (workers on their mailbox, the caller and leaders on latches) spins
// for a bounded number of iterations and then sleeps on a futex.
//
//===----------------------------------------------------------------------===//

#include "triton_cpu_rt.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <string>
#include <tuple>
#include <vector>

#if defined(__linux__)
#include <linux/futex.h>
#include <pthread.h>
#include <sched.h>
#include <signal.h>
#include <sys/syscall.h>
#include <unistd.h>
#else
#include <condition_variable>
#include <mutex>
#include <thread>
#endif

namespace {

//===----------------------------------------------------------------------===//
// Small platform helpers
//===----------------------------------------------------------------------===//

int64_t nowNs() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int currentCpu() {
#if defined(__linux__)
  return sched_getcpu();
#else
  return -1;
#endif
}

int64_t osThreadId() {
#if defined(__linux__)
  return static_cast<int64_t>(syscall(SYS_gettid));
#else
  return 0;
#endif
}

inline void cpuRelax() {
#if defined(__x86_64__) || defined(__i386__)
  __builtin_ia32_pause();
#elif defined(__aarch64__)
  asm volatile("yield" ::: "memory");
#elif defined(__riscv)
  // Zihintpause `pause` (fence w, 0).  Cores without the extension execute it
  // as a plain hint, so it is safe everywhere.
  asm volatile(".4byte 0x0100000f" ::: "memory");
#else
  std::atomic_signal_fence(std::memory_order_seq_cst);
#endif
}

int envInt(const char *name, int fallback) {
  const char *value = std::getenv(name);
  if (value == nullptr || value[0] == '\0')
    return fallback;
  char *end = nullptr;
  long parsed = std::strtol(value, &end, 10);
  if (end == value || parsed < 0 || parsed > INT_MAX)
    return fallback;
  return static_cast<int>(parsed);
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
#if defined(__linux__)
    seq_.fetch_add(1, std::memory_order_seq_cst);
    // Pairs with the seq_cst increment of waiters_ in wait(): either the
    // waiter sees the new epoch before sleeping, or we see the waiter here.
    if (waiters_.load(std::memory_order_seq_cst) != 0)
      syscall(SYS_futex, reinterpret_cast<uint32_t *>(&seq_),
              FUTEX_WAKE_PRIVATE, INT_MAX, nullptr, nullptr, 0);
#else
    {
      std::lock_guard<std::mutex> guard(mutex_);
      seq_.fetch_add(1, std::memory_order_seq_cst);
    }
    cv_.notify_all();
#endif
  }

  // Blocks until epoch() != seen; spins `spins` times before sleeping.
  void wait(uint32_t seen, uint32_t spins) {
    for (uint32_t i = 0; i < spins; ++i) {
      if (seq_.load(std::memory_order_acquire) != seen)
        return;
      cpuRelax();
    }
#if defined(__linux__)
    waiters_.fetch_add(1, std::memory_order_seq_cst);
    while (seq_.load(std::memory_order_seq_cst) == seen)
      syscall(SYS_futex, reinterpret_cast<uint32_t *>(&seq_),
              FUTEX_WAIT_PRIVATE, seen, nullptr, nullptr, 0);
    waiters_.fetch_sub(1, std::memory_order_relaxed);
#else
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock,
             [&] { return seq_.load(std::memory_order_acquire) != seen; });
#endif
  }

private:
  std::atomic<uint32_t> seq_{0};
#if defined(__linux__)
  std::atomic<uint32_t> waiters_{0};
#else
  std::mutex mutex_;
  std::condition_variable cv_;
#endif
};

// Count-down latch.  wait() returns once arrive() was called `count` times;
// everything written by the arriving threads is visible to the waiter.
//
// A latch is reused from one round to the next, and the last arriving thread
// may still be inside done_.notify() when the waiter of the round already
// returned.  Latches therefore live as long as the pool, never on a stack
// frame of a launch.  A late notify() only causes a spurious wake-up of the
// next round's waiter, which re-checks the count.
class Latch {
public:
  void reset(int count) { count_.store(count, std::memory_order_relaxed); }

  void arrive() {
    if (count_.fetch_sub(1, std::memory_order_acq_rel) == 1)
      done_.notify();
  }

  void wait(uint32_t spins) {
    for (;;) {
      const uint32_t seen = done_.epoch();
      if (count_.load(std::memory_order_acquire) == 0)
        return;
      done_.wait(seen, spins);
    }
  }

private:
  std::atomic<int> count_{0};
  Event done_;
};

//===----------------------------------------------------------------------===//
// Topology
//===----------------------------------------------------------------------===//

// Parses a Linux CPU list such as "0-3,8,10-11".
std::vector<int> parseCpuList(const std::string &text) {
  std::vector<int> cpus;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t end = text.find(',', pos);
    if (end == std::string::npos)
      end = text.size();
    const std::string item = text.substr(pos, end - pos);
    pos = end + 1;
    if (item.empty())
      continue;
    char *rest = nullptr;
    const long first = std::strtol(item.c_str(), &rest, 10);
    if (rest == item.c_str() || first < 0)
      continue;
    long last = first;
    if (*rest == '-') {
      const char *second = rest + 1;
      last = std::strtol(second, &rest, 10);
      if (rest == second || last < first)
        continue;
    }
    for (long cpu = first; cpu <= last && cpu < (1L << 20); ++cpu)
      cpus.push_back(static_cast<int>(cpu));
  }
  std::sort(cpus.begin(), cpus.end());
  cpus.erase(std::unique(cpus.begin(), cpus.end()), cpus.end());
  return cpus;
}

std::vector<int> availableCpus() {
  std::vector<int> cpus;
#if defined(__linux__)
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu)
      if (CPU_ISSET(cpu, &set))
        cpus.push_back(cpu);
  }
#else
  const unsigned count = std::thread::hardware_concurrency();
  for (unsigned cpu = 0; cpu < count; ++cpu)
    cpus.push_back(static_cast<int>(cpu));
#endif
  if (cpus.empty())
    cpus.push_back(0);
  return cpus;
}

#if defined(__linux__)
bool readFirstLine(const std::string &path, std::string &line) {
  std::ifstream file(path);
  if (!file)
    return false;
  std::getline(file, line);
  return true;
}

// CPUs sharing the L2 cache of `cpu`, falling back to its cluster.
std::vector<int> sysfsDomainOf(int cpu) {
  const std::string base =
      "/sys/devices/system/cpu/cpu" + std::to_string(cpu) + "/";
  for (int index = 0;; ++index) {
    const std::string cache = base + "cache/index" + std::to_string(index);
    std::string level;
    if (!readFirstLine(cache + "/level", level))
      break;
    std::string shared;
    if (level == "2" && readFirstLine(cache + "/shared_cpu_list", shared))
      return parseCpuList(shared);
  }
  std::string cluster;
  if (readFirstLine(base + "topology/cluster_cpus_list", cluster))
    return parseCpuList(cluster);
  return {};
}
#endif

// Groups the available CPUs into domains.  TRITON_CPU_CLUSTERS ("0-3;4-7")
// overrides the detection; CPUs it does not mention form their own domains.
std::vector<std::vector<int>> detectDomains(const std::vector<int> &cpus) {
  std::vector<std::vector<int>> domains;
  std::vector<bool> assigned(cpus.size(), false);
  auto addDomain = [&](const std::vector<int> &candidate) {
    std::vector<int> domain;
    for (size_t i = 0; i < cpus.size(); ++i) {
      if (!assigned[i] &&
          std::binary_search(candidate.begin(), candidate.end(), cpus[i])) {
        domain.push_back(cpus[i]);
        assigned[i] = true;
      }
    }
    if (!domain.empty())
      domains.push_back(domain);
  };

  if (const char *env = std::getenv("TRITON_CPU_CLUSTERS")) {
    std::string spec(env);
    size_t pos = 0;
    while (pos <= spec.size()) {
      size_t end = spec.find(';', pos);
      if (end == std::string::npos)
        end = spec.size();
      addDomain(parseCpuList(spec.substr(pos, end - pos)));
      pos = end + 1;
    }
  }
#if defined(__linux__)
  for (size_t i = 0; i < cpus.size(); ++i)
    if (!assigned[i])
      addDomain(sysfsDomainOf(cpus[i]));
#endif
  // Whatever is left (no override, no sysfs information) is one domain.
  std::vector<int> rest;
  for (size_t i = 0; i < cpus.size(); ++i)
    if (!assigned[i])
      rest.push_back(cpus[i]);
  if (!rest.empty())
    addDomain(rest);
  return domains;
}

// Number of slots of size min(warps, domain size) that fit into `domains`.
int countSlots(const std::vector<std::vector<int>> &domains, int warps) {
  int slots = 0;
  for (const std::vector<int> &domain : domains) {
    const int size = static_cast<int>(domain.size());
    if (size > 0)
      slots += size / std::min(warps, size);
  }
  return slots;
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
  // Owned by the pool, not by the job (see Latch).  The job itself lives on
  // the stack of tcrt_launch().
  Latch *done = nullptr;
  TraceEntry *trace = nullptr;
  // Region records of instances executed on the calling thread.
  std::vector<RegionTraceEntry> callerRegionTrace;
};

// A parallel region posted by a slot leader to its helpers.  Lives on the
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

enum class MessageKind { Grid, Region };

struct alignas(64) Worker {
  int cpu = -1;
  uint32_t spins = 0;
  Event mail;
  // Message, written by the sender before mail.notify() and read by the
  // worker after it observed the new epoch.
  MessageKind kind = MessageKind::Grid;
  GridJob *job = nullptr;    // Grid
  int rank = 0;              // Grid: rank of the slot within the launch
  Slot *slot = nullptr;      // Grid: slot led by this worker
  RegionTask *task = nullptr; // Region
  int member = 0;            // Region: index of this worker in the slot
  // Region records of this worker during the current launch.  Cleared by the
  // dispatcher before the launch, read by it after the launch.
  std::vector<RegionTraceEntry> regionTrace;
};

// CPUs of one domain that run one program instance at a time.
struct Slot {
  std::vector<Worker *> members; // members[0] is the leader
  Latch regionDone;
};

struct Pool {
  std::vector<int> cpus;
  std::vector<std::vector<int>> domains;
  uint32_t spins = 0;
  bool started = false;
  Latch gridDone;
  std::vector<Worker *> workers;                 // started workers only
  std::vector<std::vector<Worker *>> byDomain;   // started workers per domain
  // Slot layouts per number of warps, in launch order: round robin over the
  // domains, so that a launch with fewer instances than slots spreads over
  // all clusters.  Built on first use and kept for the lifetime of the pool.
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
void runInstances(GridJob &job, int rank) {
  const int64_t gridYZ = static_cast<int64_t>(job.gridY) * job.gridZ;
  for (;;) {
    const int64_t id = job.next.fetch_add(1, std::memory_order_relaxed);
    if (id >= job.total)
      return;
    // Same flattening as the former serial loop nest: x outer, z inner.
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
    job.fn(job.args, x, y, z);
    if (entry != nullptr)
      entry->endNs = nowNs();
  }
}

// Thread local state of a thread that executes instances of a launch.
struct InstanceScope {
  InstanceScope(Slot *slot, bool tracing,
                std::vector<RegionTraceEntry> *sink)
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
    self->mail.wait(seen, self->spins);
    seen = self->mail.epoch();
    if (self->kind == MessageKind::Grid) {
      GridJob *job = self->job;
      Latch *done = job->done;
      {
        InstanceScope scope(self->slot, job->trace != nullptr,
                            &self->regionTrace);
        runInstances(*job, self->rank);
      }
      // `job` may be gone as soon as the latch opens; do not touch it after.
      done->arrive();
    } else {
      RegionTask *task = self->task;
      Latch *done = task->done;
      const int member = self->member;
      tlsInRegion = true;
      tlsRegionSink = &self->regionTrace;
      runRegionShare(*task, member);
      tlsInRegion = false;
      // `task` lives on the leader's stack; do not touch it after.
      done->arrive();
    }
  }
  return nullptr;
}

std::atomic<bool> warnedPin{false};
std::atomic<bool> warnedThread{false};

// Starts a detached worker pinned to `worker->cpu`.  Asynchronous signals are
// blocked in the worker so that they keep being delivered to the threads of
// the embedding application (e.g. the Python main thread).
bool startWorker(Worker *worker) {
#if defined(__linux__)
  sigset_t all, previous;
  sigfillset(&all);
  pthread_sigmask(SIG_SETMASK, &all, &previous);

  pthread_attr_t attr;
  pthread_attr_init(&attr);
  pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(worker->cpu, &set);
  const bool pinRequested =
      pthread_attr_setaffinity_np(&attr, sizeof(set), &set) == 0;
  pthread_t thread;
  int rc = pthread_create(&thread, &attr, workerMain, worker);
  if (rc != 0 && pinRequested) {
    // Pinning may be refused (e.g. restricted cpusets); run unpinned.
    warnOnce(warnedPin, "could not pin worker threads to their CPU");
    pthread_attr_destroy(&attr);
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_DETACHED);
    rc = pthread_create(&thread, &attr, workerMain, worker);
  }
  pthread_attr_destroy(&attr);
  pthread_sigmask(SIG_SETMASK, &previous, nullptr);
  return rc == 0;
#else
  try {
    std::thread(workerMain, worker).detach();
    return true;
  } catch (...) {
    return false;
  }
#endif
}

void startPool(Pool &pool) {
  pool.started = true;
  pool.byDomain.assign(pool.domains.size(), {});
  for (size_t d = 0; d < pool.domains.size(); ++d) {
    for (int cpu : pool.domains[d]) {
      Worker *worker = new Worker();
      worker->cpu = cpu;
      worker->spins = pool.spins;
      if (!startWorker(worker)) {
        delete worker;
        warnOnce(warnedThread, "could not create all worker threads");
        continue;
      }
      pool.workers.push_back(worker);
      pool.byDomain[d].push_back(worker);
    }
  }
}

// Slots of a launch with `warps` threads per instance, in launch order.
// Requires a started pool.
const std::vector<Slot *> &getLayout(Pool &pool, int warps) {
  auto found = pool.layouts.find(warps);
  if (found != pool.layouts.end())
    return found->second;

  // Cut every domain into consecutive slots of min(warps, size) workers; the
  // remainder of a domain (size % slot size) stays idle.
  std::vector<std::vector<Slot *>> perDomain(pool.byDomain.size());
  for (size_t d = 0; d < pool.byDomain.size(); ++d) {
    const std::vector<Worker *> &workers = pool.byDomain[d];
    if (workers.empty())
      continue;
    const size_t size = std::min<size_t>(warps, workers.size());
    for (size_t first = 0; first + size <= workers.size(); first += size) {
      Slot *slot = new Slot();
      slot->members.assign(workers.begin() + first,
                           workers.begin() + first + size);
      perDomain[d].push_back(slot);
    }
  }
  std::vector<Slot *> layout;
  for (size_t i = 0;; ++i) {
    bool any = false;
    for (const std::vector<Slot *> &slots : perDomain) {
      if (i < slots.size()) {
        layout.push_back(slots[i]);
        any = true;
      }
    }
    if (!any)
      break;
  }
  return pool.layouts.emplace(warps, std::move(layout)).first->second;
}

//===----------------------------------------------------------------------===//
// Global state
//===----------------------------------------------------------------------===//

// Serializes launches and pool (re)initialization.  Launchers currently hold
// the GIL for the whole launch, the lock covers callers that do not.
#if defined(__linux__)
pthread_mutex_t gLock = PTHREAD_MUTEX_INITIALIZER;
void lockGlobal() { pthread_mutex_lock(&gLock); }
void unlockGlobal() { pthread_mutex_unlock(&gLock); }
#else
std::mutex gLock;
void lockGlobal() { gLock.lock(); }
void unlockGlobal() { gLock.unlock(); }
#endif

Pool *gPool = nullptr;

#if defined(__linux__)
// After fork() only the forking thread exists in the child: forget the pool
// (its workers are gone) and rebuild it on first use.  The old pool is leaked
// on purpose: it describes threads that do not exist in the child.
void atforkPrepare() { lockGlobal(); }
void atforkParent() { unlockGlobal(); }
void atforkChild() {
  gPool = nullptr;
  pthread_mutex_init(&gLock, nullptr);
}
#endif

// Requires gLock.
Pool &getPoolLocked() {
  if (gPool == nullptr) {
#if defined(__linux__)
    static bool atforkRegistered = false;
    if (!atforkRegistered) {
      pthread_atfork(atforkPrepare, atforkParent, atforkChild);
      atforkRegistered = true;
    }
#endif
    Pool *pool = new Pool();
    pool->cpus = availableCpus();
    pool->domains = detectDomains(pool->cpus);
    pool->spins = static_cast<uint32_t>(envInt("TRITON_CPU_SPIN", 4000));
    gPool = pool;
  }
  return *gPool;
}

struct GlobalGuard {
  GlobalGuard() { lockGlobal(); }
  ~GlobalGuard() { unlockGlobal(); }
  GlobalGuard(const GlobalGuard &) = delete;
  GlobalGuard &operator=(const GlobalGuard &) = delete;
};

void writeTrace(const char *path, const std::vector<TraceEntry> &entries) {
  FILE *file = std::fopen(path, "w");
  if (file == nullptr)
    return;
  std::fputs("instance,start_ns,end_ns,thread_id,cpu,os_tid\n", file);
  for (size_t i = 0; i < entries.size(); ++i)
    std::fprintf(file, "%zu,%lld,%lld,%d,%d,%lld\n", i,
                 static_cast<long long>(entries[i].startNs),
                 static_cast<long long>(entries[i].endNs),
                 entries[i].threadId, entries[i].cpu,
                 static_cast<long long>(entries[i].osTid));
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
    std::fprintf(file, "%lld,%d,%lld,%lld,%lld,%d,%lld,%lld\n",
                 static_cast<long long>(e.instance), e.region,
                 static_cast<long long>(e.tid),
                 static_cast<long long>(e.nthreads),
                 static_cast<long long>(e.osTid), e.cpu,
                 static_cast<long long>(e.startNs),
                 static_cast<long long>(e.endNs));
  std::fclose(file);
}

} // namespace

//===----------------------------------------------------------------------===//
// ABI
//===----------------------------------------------------------------------===//

extern "C" TCRT_API int tcrt_launch(const tcrt_launch_desc *desc) {
  if (desc == nullptr || desc->abi_version != TCRT_ABI_VERSION ||
      desc->fn == nullptr)
    return -1;
  if (desc->grid[0] <= 0 || desc->grid[1] <= 0 || desc->grid[2] <= 0)
    return 0;
  const int warps = std::max(desc->warps, 1);

  GridJob job;
  job.fn = desc->fn;
  job.args = desc->args;
  job.gridY = desc->grid[1];
  job.gridZ = desc->grid[2];
  job.total = static_cast<int64_t>(desc->grid[0]) * desc->grid[1] *
              desc->grid[2];

  const bool tracing = desc->trace_path != nullptr && desc->trace_path[0];
  std::vector<TraceEntry> trace(tracing ? static_cast<size_t>(job.total) : 0);
  job.trace = tracing ? trace.data() : nullptr;

  auto runOnCaller = [&] {
    InstanceScope scope(nullptr, tracing, &job.callerRegionTrace);
    runInstances(job, 0);
  };

  std::vector<RegionTraceEntry> regionTrace;
  if (tlsWorker != nullptr) {
    // A launch issued from a pool worker (nested launch) runs inline: the
    // pool is busy with the enclosing launch.
    runOnCaller();
  } else {
    GlobalGuard guard;
    Pool &pool = getPoolLocked();

    const int requested = envInt("TRITON_CPU_NUM_THREADS", 0);
    int64_t limit = job.total;
    if (requested > 0)
      limit = std::min<int64_t>(limit, requested);

    // A single instance at a time on a single CPU: no pool round trip.
    const bool single =
        warps == 1 &&
        std::min<int64_t>(limit, static_cast<int64_t>(pool.cpus.size())) <= 1;
    if (!single && !pool.started)
      startPool(pool);
    const std::vector<Slot *> *layout =
        single ? nullptr : &getLayout(pool, warps);
    const int64_t parallel =
        layout == nullptr
            ? 0
            : std::min<int64_t>(limit, static_cast<int64_t>(layout->size()));

    if (parallel == 0 ||
        (parallel == 1 && (*layout)[0]->members.size() == 1)) {
      runOnCaller();
    } else {
      const int leaders = static_cast<int>(parallel);
      job.done = &pool.gridDone;
      job.done->reset(leaders);
      if (tracing)
        for (int rank = 0; rank < leaders; ++rank)
          for (Worker *member : (*layout)[rank]->members)
            member->regionTrace.clear();
      for (int rank = 0; rank < leaders; ++rank) {
        Slot *slot = (*layout)[rank];
        Worker *leader = slot->members[0];
        leader->kind = MessageKind::Grid;
        leader->job = &job;
        leader->rank = rank;
        leader->slot = slot;
        leader->mail.notify();
      }
      job.done->wait(pool.spins);
      if (tracing)
        for (int rank = 0; rank < leaders; ++rank)
          for (Worker *member : (*layout)[rank]->members)
            regionTrace.insert(regionTrace.end(), member->regionTrace.begin(),
                               member->regionTrace.end());
    }
  }

  if (tracing) {
    writeTrace(desc->trace_path, trace);
    regionTrace.insert(regionTrace.end(), job.callerRegionTrace.begin(),
                       job.callerRegionTrace.end());
    writeRegionTrace(desc->trace_path, regionTrace);
  }
  return 0;
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
  task.done->wait(slot->members[0]->spins);
}

extern "C" TCRT_API int32_t triton_cpu_current_cpu(void) {
  return static_cast<int32_t>(currentCpu());
}

extern "C" TCRT_API int32_t triton_cpu_os_thread_id(void) {
  return static_cast<int32_t>(osThreadId());
}

extern "C" TCRT_API int tcrt_num_cpus(void) {
  GlobalGuard guard;
  return static_cast<int>(getPoolLocked().cpus.size());
}

extern "C" TCRT_API int tcrt_num_domains(void) {
  GlobalGuard guard;
  return static_cast<int>(getPoolLocked().domains.size());
}

extern "C" TCRT_API int tcrt_domain_cpus(int domain, int *cpus, int cap) {
  GlobalGuard guard;
  const Pool &pool = getPoolLocked();
  if (domain < 0 || static_cast<size_t>(domain) >= pool.domains.size())
    return -1;
  const std::vector<int> &list = pool.domains[domain];
  for (int i = 0; i < cap && static_cast<size_t>(i) < list.size(); ++i)
    cpus[i] = list[i];
  return static_cast<int>(list.size());
}

extern "C" TCRT_API int tcrt_num_slots(int warps) {
  GlobalGuard guard;
  Pool &pool = getPoolLocked();
  warps = std::max(warps, 1);
  if (pool.started)
    return static_cast<int>(getLayout(pool, warps).size());
  return countSlots(pool.domains, warps);
}
