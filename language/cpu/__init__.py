"""CPU specific Triton language extensions of triton-riscv.

Installed as ``triton.language.extra.cpu`` (Triton links the ``language``
directory of external backends into ``triton.language.extra``).

Example::

    from triton.language.extra import cpu

    @triton.jit
    def kernel(x_ptr, out_ptr, n, BLOCK: tl.constexpr):
        ...                                   # executed by one thread
        for i in cpu.parallel(0, n, BLOCK):   # executed by num_warps threads
            offs = i + tl.arange(0, BLOCK)
            ...
        ...                                   # one thread again

    kernel[grid](..., num_warps=4)

Inside a kernel, ``cpu.current_cpu()`` and ``cpu.os_thread_id()`` return the
CPU and the Linux thread id of the thread executing the call, and
``cpu.alloc(shape, dtype)`` returns a pointer to a scratch buffer in the TCM of
that thread.
"""

from .introspection import current_cpu, os_thread_id
from .memory import alloc
from .parallel import parallel

__all__ = ["alloc", "current_cpu", "os_thread_id", "parallel"]
