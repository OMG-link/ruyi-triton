//===----------------------------------------------------------------------===//
//
// Lowers the buffers of triton.language.extra.cpu.alloc to memref.alloc ops
// marked `cpu.tcm`.  See CpuTcmAllocToMemrefPass.cpp.
//
//===----------------------------------------------------------------------===//

#ifndef TRITON_CONVERSION_TRITONTOLINALG_CPUTCMALLOCTOMEMREF_H
#define TRITON_CONVERSION_TRITONTOLINALG_CPUTCMALLOCTOMEMREF_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createCpuTcmAllocToMemrefPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_CONVERSION_TRITONTOLINALG_CPUTCMALLOCTOMEMREF_H
