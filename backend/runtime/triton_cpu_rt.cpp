//===- triton_cpu_rt.cpp - triton-riscv CPU runtime -------------*- C++ -*-===//
//
// Worker pool that executes Triton launch grids on the CPU.  See
// triton_cpu_rt.h for the ABI, docs/05-Program-Instance-Parallelism.md and
// docs/06-Program-Instance-Threads.md for the design.
//
// Structure:
//   * Topology: the CPUs available to the process (sched_getaffinity) grouped
//     into "domains", i.e. CPUs sharing an L2 cache (one cluster on the K1).
//   * AI cores (SpacemiT K3): the A100 cores are outside the affinity of
//     ordinary threads; a thread gets access to them, and loses access to the
//     other cores, by writing its id to /proc/set_ai_thread.  When that file
//     exists (and TRITON_CPU_AI_CORES is not "0") the pool uses the AI cores
//     only: a short lived thread enables them for itself, reads the resulting
//     affinity, and creates the workers, which inherit the access and are
//     pinned to one AI core each.  No kernel code then runs on any other core,
//     not even on the calling thread.
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
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <functional>
#include <map>
#include <mutex>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#if defined(__linux__)
#include <dlfcn.h>
#include <fcntl.h>
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

//===----------------------------------------------------------------------===//
// AI cores (SpacemiT K3)
//===----------------------------------------------------------------------===//

#if defined(__linux__)
const char *const kAiThreadPath = "/proc/set_ai_thread";
#endif

// True if the pool should run on the AI cores: the kernel interface exists
// and TRITON_CPU_AI_CORES does not opt out.
bool aiCoresRequested() {
#if defined(__linux__)
  const char *env = std::getenv("TRITON_CPU_AI_CORES");
  if (env != nullptr && std::strcmp(env, "0") == 0)
    return false;
  return access(kAiThreadPath, W_OK) == 0;
#else
  return false;
#endif
}

#if defined(__linux__)
// Moves the calling thread to the AI cores.  The move is one way: afterwards
// the thread (and every thread it creates) can only run on AI cores.  The
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
#endif

// Runs `body` on a new, short lived thread that has been moved to the AI cores
// and waits for it.  The calling thread is left untouched.  Returns false (and
// does not run `body`) if the thread could not be created or moved.
bool runOnAiThread(const std::function<void()> &body) {
#if defined(__linux__)
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
#else
  (void)body;
  return false;
#endif
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

enum class MessageKind { Grid, Region, TcmRelease };

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
  Latch *released = nullptr; // TcmRelease
  // Region records of this worker during the current launch.  Cleared by the
  // dispatcher before the launch, read by it after the launch.
  std::vector<RegionTraceEntry> regionTrace;
  // cpu.alloc buffers (see "TCM" below), set up by the worker itself on first
  // use: `tcm` is the TCM block `tcmBlock` owned by the worker, or ordinary
  // memory (`tcmBlock` < 0).  Only the worker writes these.
  bool onAiCore = false; // pinned to an AI core (pool in AI mode)
  bool tcmReady = false;
  int tcmBlock = -1;
  void *tcm = nullptr;
  size_t tcmBytes = 0; // usable size of `tcm`
};

// CPUs of one domain that run one program instance at a time.
struct Slot {
  std::vector<Worker *> members; // members[0] is the leader
  Latch regionDone;
  // The leader checked that the slot's CPUs are next to its TCM block (see
  // triton_cpu_tcm_base_shared).  Only the leader touches it.
  bool tcmChecked = false;
};

struct Pool {
  std::vector<int> cpus;
  std::vector<std::vector<int>> domains;
  uint32_t spins = 0;
  // The pool runs on the AI cores (`cpus` are AI cores); kernel code must
  // then never run on the calling thread.
  bool ai = false;
  bool started = false;
  Latch gridDone;
  std::vector<Worker *> workers;                 // started workers only
  std::vector<std::vector<Worker *>> byDomain;   // started workers per domain
  // Slot layouts per number of warps, in launch order: round robin over the
  // domains, so that a launch with fewer instances than slots spreads over
  // all clusters.  Built on first use and kept for the lifetime of the pool.
  std::map<int, std::vector<Slot *>> layouts;
  // Counts the workers that released their TCM block at exit.
  Latch tcmReleased;
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
// TCM (SpacemiT K3): buffers of triton.language.extra.cpu.alloc
//===----------------------------------------------------------------------===//
//
// Compiled kernels get the base of their cpu.alloc buffers from
// triton_cpu_tcm_base(bytes) (buddy-mlir --cpu-tcm-assign), called by the
// thread that executes the allocation.  On the AI cores every worker uses the
// TCM block next to its core (cpu_affinity_mask), acquired exclusively with
// spine_tcm_mem_try_wait() by the worker itself on first use and released by
// it at exit (the library only accepts a release from the acquiring thread).
// Everywhere else, and when the block is taken by another process, the
// buffers are in ordinary memory of the thread.  TRITON_CPU_TCM=0 always uses
// ordinary memory, TRITON_CPU_TCM=require makes launches fail instead.
//
// libspine_tcm is loaded with dlopen, so the runtime does not depend on it.
// The X100 cores do not write the TCM reliably and are never given a block.

constexpr size_t kTcmAlign = 128;
constexpr size_t kDefaultTcmBytes = 384 * 1024;

enum class TcmMode { Off, Auto, Require };

TcmMode tcmMode() {
  const char *env = std::getenv("TRITON_CPU_TCM");
  if (env != nullptr && std::strcmp(env, "0") == 0)
    return TcmMode::Off;
  if (env != nullptr && std::strcmp(env, "require") == 0)
    return TcmMode::Require;
  return TcmMode::Auto;
}

// Set when a buffer could not be put in the TCM with TRITON_CPU_TCM=require;
// reported by tcrt_launch.
std::atomic<bool> gTcmRequireFailed{false};

std::atomic<bool> warnedTcmLib{false};
std::atomic<bool> warnedTcmBusy{false};
std::atomic<bool> warnedTcmStale{false};
std::atomic<bool> warnedTcmSize{false};
std::atomic<bool> warnedTcmShared{false};

void warnOncef(std::atomic<bool> &flag, const char *format, ...) {
  if (flag.exchange(true))
    return;
  char message[512];
  va_list args;
  va_start(args, format);
  std::vsnprintf(message, sizeof(message), format, args);
  va_end(args);
  std::fprintf(stderr, "triton-cpu-rt: warning: %s\n", message);
}

// Mirrors of the structs of spine_tcm.h (spacemit-tcm 3.x).
struct SpineTcmMemInfo {
  size_t blkSize;
  size_t blkNum;
  int isFakeTcm;
};
struct SpineTcmBlockInfo {
  int id;
  void *va;
  size_t size;
  uint64_t physAddr;
  uint64_t cpuAffinityMask;
  int ownerTid;
  int isAcquired;
};

struct TcmLib {
  bool ok = false;
  int (*isAvailable)(void) = nullptr;
  int (*layoutInfo)(SpineTcmMemInfo *) = nullptr;
  int (*memInfo)(int, SpineTcmBlockInfo *) = nullptr;
  void *(*tryWait)(int, size_t) = nullptr;
  int (*release)(int) = nullptr;
  int (*forceRelease)(int) = nullptr;
  size_t blockBytes = 0;
  std::vector<SpineTcmBlockInfo> blocks;
};

// The library, loaded on first use.
const TcmLib &tcmLib() {
  static TcmLib lib;
  static std::once_flag once;
  std::call_once(once, [] {
#if defined(__linux__)
    void *handle = dlopen("libspine_tcm.so", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr)
      handle = dlopen("libspine_tcm.so.3", RTLD_NOW | RTLD_LOCAL);
    if (handle == nullptr)
      return;
    auto sym = [&](auto &fn, const char *name) {
      fn = reinterpret_cast<std::remove_reference_t<decltype(fn)>>(
          dlsym(handle, name));
      return fn != nullptr;
    };
    if (!sym(lib.isAvailable, "spine_tcm_runtime_is_available") ||
        !sym(lib.layoutInfo, "spine_tcm_runtime_layout_info") ||
        !sym(lib.memInfo, "spine_tcm_runtime_mem_info") ||
        !sym(lib.tryWait, "spine_tcm_runtime_mem_try_wait") ||
        !sym(lib.release, "spine_tcm_runtime_mem_release") ||
        !sym(lib.forceRelease, "spine_tcm_runtime_mem_force_release"))
      return;
    SpineTcmMemInfo info{};
    if (lib.isAvailable() != 1 || lib.layoutInfo(&info) != 0 ||
        info.isFakeTcm != 0 || info.blkNum == 0 || info.blkSize == 0)
      return;
    for (size_t id = 0; id < info.blkNum; ++id) {
      SpineTcmBlockInfo block{};
      if (lib.memInfo(static_cast<int>(id), &block) != 0)
        return;
      lib.blocks.push_back(block);
    }
    lib.blockBytes = info.blkSize;
    lib.ok = true;
#endif
  });
  return lib;
}

// Block of the TCM next to `cpu`: among the blocks whose affinity mask
// contains `cpu`, the one at the rank of `cpu` within the mask (on the K3 two
// cores share two blocks: the lower core gets the lower block).
int tcmBlockFor(const TcmLib &lib, int cpu) {
  if (cpu < 0 || cpu >= 64)
    return -1;
  std::vector<int> candidates;
  uint64_t mask = 0;
  for (const SpineTcmBlockInfo &block : lib.blocks) {
    if ((block.cpuAffinityMask >> cpu) & 1) {
      candidates.push_back(block.id);
      mask = block.cpuAffinityMask;
    }
  }
  if (candidates.empty())
    return -1;
  int rank = 0;
  for (int lower = 0; lower < cpu; ++lower)
    rank += static_cast<int>((mask >> lower) & 1);
  return candidates[static_cast<size_t>(rank) % candidates.size()];
}

uint64_t tcmBlockMask(int block) {
  const TcmLib &lib = tcmLib();
  if (block < 0 || static_cast<size_t>(block) >= lib.blocks.size())
    return 0;
  return lib.blocks[block].cpuAffinityMask;
}

bool threadExists(int tid) {
#if defined(__linux__)
  // /proc/<tid> exists for every thread, not only for thread group leaders.
  return tid > 0 && access(("/proc/" + std::to_string(tid)).c_str(), F_OK) == 0;
#else
  (void)tid;
  return true;
#endif
}

void *allocAligned(size_t bytes) {
  bytes = (bytes + kTcmAlign - 1) / kTcmAlign * kTcmAlign;
#if defined(_WIN32)
  return _aligned_malloc(bytes, kTcmAlign);
#else
  return std::aligned_alloc(kTcmAlign, bytes);
#endif
}

void freeAligned(void *p) {
#if defined(_WIN32)
  _aligned_free(p);
#else
  std::free(p);
#endif
}

// Ordinary memory standing in for the TCM of a thread: at least `bytes`
// (and the size of a TCM block), grown on demand.  Grows only between program
// instances: every call of one kernel asks for the same size.
struct FallbackBuffer {
  void *data = nullptr;
  size_t bytes = 0;
  ~FallbackBuffer() { freeAligned(data); }
  void *get(size_t needed) {
    needed = std::max(needed, kDefaultTcmBytes);
    if (needed > bytes) {
      freeAligned(data);
      data = allocAligned(needed);
      if (data == nullptr) {
        std::fprintf(stderr, "triton-cpu-rt: error: cannot allocate %zu bytes "
                             "for cpu.alloc buffers\n",
                     needed);
        std::abort();
      }
      bytes = needed;
    }
    return data;
  }
};

void registerTcmReleaseAtExit(); // see "Global state"

// Acquires the TCM block of `worker`; on the worker's own thread.
void acquireTcm(Worker *worker) {
  worker->tcmReady = true;
  const TcmMode mode = tcmMode();
  if (mode == TcmMode::Off || !worker->onAiCore)
    return;
  const TcmLib &lib = tcmLib();
  if (!lib.ok) {
    warnOnce(warnedTcmLib, "the TCM (libspine_tcm) is not available; "
                           "cpu.alloc buffers are in ordinary memory");
    return;
  }
  const int block = tcmBlockFor(lib, worker->cpu);
  if (block < 0)
    return;
  void *base = lib.tryWait(block, 0);
  if (base == nullptr) {
    // A block whose owner thread no longer exists was left behind by a
    // process that died while holding it.
    SpineTcmBlockInfo info{};
    if (lib.memInfo(block, &info) == 0 && info.isAcquired != 0 &&
        info.ownerTid > 0 && !threadExists(info.ownerTid)) {
      warnOncef(warnedTcmStale,
                "reclaiming TCM block %d left behind by thread %d, which no "
                "longer exists",
                block, info.ownerTid);
      lib.forceRelease(block);
      base = lib.tryWait(block, 0);
    }
  }
  if (base == nullptr) {
    SpineTcmBlockInfo info{};
    lib.memInfo(block, &info);
    warnOncef(warnedTcmBusy,
              "TCM block %d (cpu %d) is used by thread %d; cpu.alloc buffers "
              "of this thread are in ordinary memory",
              block, worker->cpu, info.ownerTid);
    return;
  }
  worker->tcmBlock = block;
  worker->tcm = base;
  worker->tcmBytes = lib.blockBytes;
  registerTcmReleaseAtExit();
}

// Releases the TCM block of `worker`; on the worker's own thread.  The next
// use acquires it again.
void releaseTcm(Worker *worker) {
  if (worker->tcmBlock >= 0)
    tcmLib().release(worker->tcmBlock);
  worker->tcmReady = false;
  worker->tcmBlock = -1;
  worker->tcm = nullptr;
  worker->tcmBytes = 0;
}

thread_local FallbackBuffer tlsTcmFallback;

// Base of the cpu.alloc buffers (`bytes` in total) of the calling thread.
void *tcmBase(int64_t bytes) {
  const size_t needed = bytes > 0 ? static_cast<size_t>(bytes) : 0;
  Worker *worker = tlsWorker;
  if (worker != nullptr) {
    if (!worker->tcmReady)
      acquireTcm(worker);
    if (worker->tcmBlock >= 0) {
      if (needed <= worker->tcmBytes)
        return worker->tcm;
      warnOncef(warnedTcmSize,
                "cpu.alloc buffers of %zu bytes do not fit into a TCM block "
                "of %zu bytes; using ordinary memory",
                needed, worker->tcmBytes);
    }
  }
  if (tcmMode() == TcmMode::Require)
    gTcmRequireFailed.store(true, std::memory_order_relaxed);
  return tlsTcmFallback.get(needed);
}

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
    } else if (self->kind == MessageKind::TcmRelease) {
      Latch *done = self->released;
      releaseTcm(self);
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
std::atomic<bool> warnedAi{false};
std::atomic<bool> warnedAiEmpty{false};

// Starts a detached worker pinned to `worker->cpu`.  Asynchronous signals are
// blocked in the worker so that they keep being delivered to the threads of
// the embedding application (e.g. the Python main thread).  If pinning is
// refused the worker runs unpinned, unless `mustPin` is set, in which case it
// is not started at all.
bool startWorker(Worker *worker, bool mustPin) {
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
  int rc = -1;
  if (pinRequested || !mustPin)
    rc = pthread_create(&thread, &attr, workerMain, worker);
  if (rc != 0 && pinRequested && !mustPin) {
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
  (void)mustPin;
  try {
    std::thread(workerMain, worker).detach();
    return true;
  } catch (...) {
    return false;
  }
#endif
}

void createWorkers(Pool &pool) {
  for (size_t d = 0; d < pool.domains.size(); ++d) {
    for (int cpu : pool.domains[d]) {
      Worker *worker = new Worker();
      worker->cpu = cpu;
      worker->spins = pool.spins;
      worker->onAiCore = pool.ai;
      // An unpinned AI worker could still only run on AI cores, but would
      // break the one-worker-per-core layout; leave the core out instead.
      if (!startWorker(worker, /*mustPin=*/pool.ai)) {
        delete worker;
        warnOnce(warnedThread, "could not create all worker threads");
        continue;
      }
      pool.workers.push_back(worker);
      pool.byDomain[d].push_back(worker);
    }
  }
}

void startPool(Pool &pool) {
  pool.started = true;
  pool.byDomain.assign(pool.domains.size(), {});
  if (!pool.ai) {
    createWorkers(pool);
    return;
  }
  // Workers created by a thread that has access to the AI cores inherit the
  // access, so they can be pinned to their AI core from the start and never
  // execute anything on another core.  The creating thread is joined before
  // runOnAiThread() returns, which publishes its writes to `pool`.
  if (!runOnAiThread([&pool] { createWorkers(pool); }))
    warnOnce(warnedAi, "could not move the worker threads to the AI cores");
  if (pool.workers.empty())
    warnOnce(warnedAiEmpty, "no worker thread on the AI cores; kernels run "
                            "on the calling thread");
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
    if (aiCoresRequested()) {
      // The AI cores are the affinity of a thread moved to them.  Probe on a
      // throw-away thread: the move cannot be undone.
      std::vector<int> aiCpus;
      if (runOnAiThread([&aiCpus] { aiCpus = availableCpus(); })) {
        pool->ai = true;
        pool->cpus = aiCpus;
      } else {
        warnOnce(warnedAi, "could not move threads to the AI cores "
                           "(/proc/set_ai_thread); using the CPUs of the "
                           "process");
      }
    }
    if (!pool->ai)
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

// Every worker holding a TCM block releases it (only the acquiring thread
// may); at exit and on tcrt_tcm_release().  A process killed while holding
// blocks leaves them behind; the next process reclaims them (acquireTcm).
void releaseAllTcm() {
  GlobalGuard guard;
  Pool *pool = gPool;
  if (pool == nullptr)
    return;
  std::vector<Worker *> holders;
  for (Worker *worker : pool->workers)
    if (worker->tcmBlock >= 0)
      holders.push_back(worker);
  if (holders.empty())
    return;
  pool->tcmReleased.reset(static_cast<int>(holders.size()));
  for (Worker *worker : holders) {
    worker->kind = MessageKind::TcmRelease;
    worker->released = &pool->tcmReleased;
    worker->mail.notify();
  }
  pool->tcmReleased.wait(pool->spins);
}

void registerTcmReleaseAtExit() {
  static std::once_flag once;
  std::call_once(once, [] { std::atexit(releaseAllTcm); });
}

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
  if (tlsWorker == nullptr)
    gTcmRequireFailed.store(false, std::memory_order_relaxed);

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

    // A single instance at a time on a single CPU: no pool round trip.  Not
    // on the AI cores, where kernel code must not run on the calling thread.
    const bool single =
        !pool.ai && warps == 1 &&
        std::min<int64_t>(limit, static_cast<int64_t>(pool.cpus.size())) <= 1;
    if (!single && !pool.started)
      startPool(pool);
    const std::vector<Slot *> *layout =
        single ? nullptr : &getLayout(pool, warps);
    const int64_t parallel =
        layout == nullptr
            ? 0
            : std::min<int64_t>(limit, static_cast<int64_t>(layout->size()));

    // parallel == 0 with the AI cores means that no AI worker could be
    // started (warned in startPool).
    if (parallel == 0 || (!pool.ai && parallel == 1 &&
                          (*layout)[0]->members.size() == 1)) {
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
  // TRITON_CPU_TCM=require and some cpu.alloc buffer was not in the TCM.  The
  // launch itself completed (in ordinary memory).
  if (gTcmRequireFailed.exchange(false, std::memory_order_relaxed))
    return TCRT_ERROR_NO_TCM;
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

extern "C" TCRT_API void *triton_cpu_tcm_base(int64_t bytes) {
  return tcmBase(bytes);
}

extern "C" TCRT_API void *triton_cpu_tcm_base_shared(int64_t bytes) {
  void *base = tcmBase(bytes);
  // The buffers are used by the whole slot of the calling leader: check once
  // per slot that its CPUs are next to the leader's block (only the two cores
  // of a pair are; the others access it more than 100x slower).
  Worker *worker = tlsWorker;
  Slot *slot = tlsInRegion ? nullptr : tlsSlot;
  if (worker != nullptr && slot != nullptr && worker->tcmBlock >= 0 &&
      base == worker->tcm && !slot->tcmChecked) {
    slot->tcmChecked = true;
    const uint64_t mask = tcmBlockMask(worker->tcmBlock);
    for (Worker *member : slot->members) {
      if (member->cpu >= 0 && member->cpu < 64 && ((mask >> member->cpu) & 1))
        continue;
      warnOncef(warnedTcmShared,
                "a cpu.alloc buffer used inside cpu.parallel is in the TCM "
                "of cpu %d, which cpu %d of the same program instance "
                "accesses slowly; use num_warps=2",
                worker->cpu, member->cpu);
      break;
    }
  }
  return base;
}

extern "C" TCRT_API void tcrt_tcm_release(void) { releaseAllTcm(); }

extern "C" TCRT_API int tcrt_tcm_available(void) {
  GlobalGuard guard;
  return getPoolLocked().ai && tcmMode() != TcmMode::Off && tcmLib().ok ? 1
                                                                        : 0;
}

extern "C" TCRT_API int64_t tcrt_tcm_block_bytes(void) {
  const TcmLib &lib = tcmLib();
  return lib.ok ? static_cast<int64_t>(lib.blockBytes) : 0;
}

extern "C" TCRT_API int tcrt_worker_tcm(int cpu, int *block, void **base) {
  GlobalGuard guard;
  const Pool &pool = getPoolLocked();
  for (const Worker *worker : pool.workers) {
    if (worker->cpu != cpu)
      continue;
    if (!worker->tcmReady)
      return -1;
    if (block != nullptr)
      *block = worker->tcmBlock;
    if (base != nullptr)
      *base = worker->tcm;
    return worker->tcmBlock >= 0 ? 1 : 0;
  }
  return -1;
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

extern "C" TCRT_API int tcrt_ai_cores(void) {
  GlobalGuard guard;
  return getPoolLocked().ai ? 1 : 0;
}

extern "C" TCRT_API int tcrt_num_slots(int warps) {
  GlobalGuard guard;
  Pool &pool = getPoolLocked();
  warps = std::max(warps, 1);
  if (pool.started)
    return static_cast<int>(getLayout(pool, warps).size());
  return countSlots(pool.domains, warps);
}
