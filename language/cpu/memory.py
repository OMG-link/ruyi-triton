import math

from triton.language import core

# Runtime symbol behind cpu.alloc.  triton-shared-opt (--cpu-tcm-alloc-to-memref)
# replaces the call by a memref.alloc marked `cpu.tcm`; buddy-mlir places these
# buffers in the TCM of the executing thread (see tcm-alloc-plan.md).
_TCM_ALLOC_SYMBOL = "__triton_cpu_tcm_alloc"

# Alignment of every buffer: one vector register of the SpacemiT A100
# (VLEN = 1024 bit).
_TCM_ALIGN = 128


def _static_shape(shape):
    if isinstance(shape, (int, core.constexpr)):
        shape = [shape]
    dims = core._unwrap_shape(shape)
    for dim in dims:
        if not isinstance(dim, int) or isinstance(dim, bool):
            raise ValueError(f"cpu.alloc: shape must be compile-time integers, got {shape}")
        if dim <= 0:
            raise ValueError(f"cpu.alloc: shape entries must be positive, got {dims}")
    return dims


@core.builtin
def alloc(shape, dtype, _semantic=None):
    """Allocates a scratch buffer in the TCM of the thread executing the call.

    ``shape`` (compile-time integers) and ``dtype`` give the size of the buffer:
    ``prod(shape)`` contiguous elements of ``dtype``, 128-byte aligned.  Returns
    a plain pointer (``tl.pointer_type(dtype)``) to the first element; address
    the buffer like any other pointer, e.g. row major
    ``p + rows[:, None] * N + cols[None, :]``.

    * The buffer is in the TCM of the thread that executes the call: outside of
      a ``cpu.parallel`` loop the leader of the program instance, inside it each
      thread gets its own buffer.  Buffers allocated outside of a loop can be
      used inside it by all threads of the instance; use ``num_warps=2`` so that
      both threads are next to the same TCM block.
    * Contents are undefined on allocation.  The buffer lives until the end of
      the program instance (or of the parallel loop iteration it was allocated
      in) and is never freed explicitly.
    * All ``cpu.alloc`` calls of a kernel together must fit into one TCM block
      (384 KB on the SpacemiT K3); compilation fails otherwise.
    * Atomic operations on the buffer are not supported (the TCM has no atomics).
    * Without a usable TCM (other platforms, ``TRITON_CPU_AI_CORES=0``, the
      blocks taken by another process) the buffer is in ordinary memory.
    """
    dims = _static_shape(shape)
    dtype = core._unwrap_if_constexpr(dtype)
    if not isinstance(dtype, core.dtype) or dtype.is_ptr() or dtype.is_block():
        raise ValueError(f"cpu.alloc: dtype must be a scalar type, got {dtype}")
    element_bytes = max(1, (dtype.primitive_bitwidth + 7) // 8)
    nbytes = math.prod(dims) * element_bytes

    builder = _semantic.builder
    size = core.tensor(builder.get_int64(nbytes), core.int64)
    align = core.tensor(builder.get_int64(_TCM_ALIGN), core.int64)
    address = core.extern_elementwise(
        "", "", [size, align], {(core.int64, core.int64): (_TCM_ALLOC_SYMBOL, core.int64)}, is_pure=False,
        _semantic=_semantic)
    return _semantic.cast(address, core.pointer_type(dtype))
