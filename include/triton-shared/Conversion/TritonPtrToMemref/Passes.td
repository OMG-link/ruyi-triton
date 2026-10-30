#ifndef TRITON_PTR_TO_MEMREF_CONVERSION_PASSES
#define TRITON_PTR_TO_MEMREF_CONVERSION_PASSES

include "mlir/Pass/PassBase.td"

def TritonPtrToMemref : Pass<"triton-ptr-to-memref", "mlir::ModuleOp"> {
  let summary = "Convert Triton pointer signatures to bare-pointer memrefs";
  let description = [{
    Convert scalar Triton pointer arguments and results to rank-zero memrefs
    and mark their functions with `llvm.bareptr`. FuncToLLVM then passes each
    such memref as one data pointer and reconstructs its descriptor inside the
    function, with identical allocated/aligned pointers and zero offset.
    This wrapping does not transfer ownership of the pointed-to memory.
    Internal pointer users retain the unranked memref representation.
  }];
  let constructor = "triton::createTritonPtrToMemrefPass()";
}

#endif
