from triton.language import core

# Implemented by the CPU runtime (backend/runtime/triton_cpu_rt.h); the calls
# are lowered to plain function calls by triton-shared-opt and resolved when
# the kernel is linked into its launcher.


@core.builtin
def current_cpu(_semantic=None):
    """CPU (Linux CPU number) the thread executing this code is running on.

    Read with sched_getcpu() at the time of the call, by the thread that
    executes it.  Inside a ``cpu.parallel`` loop that is the thread running the
    current iteration.  Returns an int32 scalar, -1 if unknown.
    """
    return core.extern_elementwise("", "", [], {(): ("triton_cpu_current_cpu", core.int32)}, is_pure=False,
                                   _semantic=_semantic)


@core.builtin
def os_thread_id(_semantic=None):
    """Linux thread id (gettid) of the thread executing this code (int32 scalar)."""
    return core.extern_elementwise("", "", [], {(): ("triton_cpu_os_thread_id", core.int32)}, is_pure=False,
                                   _semantic=_semantic)
