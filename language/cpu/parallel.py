from triton.language import core

# Discardable attribute understood by buddy-mlir's --cpar-distribute-loops.
PARALLEL_LOOP_ATTR = "cpar.parallel"


class parallel(core.range):
    """Loop whose iterations are executed by the threads of the program instance.

    ``for i in cpu.parallel(start, end, step)`` behaves like ``tl.range``, with
    one additional promise from the user: the iterations are independent (no
    iteration reads what another one writes, and the loop does not carry values
    from one iteration to the next).  The CPU backend then distributes the
    iterations round robin over the ``num_warps`` threads of the program
    instance (thread ``t`` runs iterations ``t, t + num_warps, ...``) and waits
    for all of them at the end of the loop.  Code outside the loop is executed
    by a single thread.

    * ``num_warps > 0`` (kernel launch option): number of threads.
    * ``num_warps = 0`` (default): the backend picks the number of threads.
    * Nested ``cpu.parallel`` loops: only the outermost one is distributed.
    * Values defined before the loop are shared by all threads.  Tensors
      created inside the loop body are private to the thread running the
      iteration.

    Accepts the same keyword arguments as ``tl.range``.
    """

    def __init__(self, arg1, arg2=None, step=None, **kwargs):
        super().__init__(arg1, arg2, step, **kwargs)

    # Called by the Triton code generator (patches/0005) once the scf.for of
    # the loop has been created.
    def _annotate_loop(self, for_op, builder):
        for_op.set_attr(PARALLEL_LOOP_ATTR, builder.get_unit_attr())
