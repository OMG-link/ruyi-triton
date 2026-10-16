// Licensed under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
#ifndef TRITON_TRANSFORM_MARKNOALIASARGS_MARKNOALIASARGS_H
#define TRITON_TRANSFORM_MARKNOALIASARGS_MARKNOALIASARGS_H

#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace triton {

std::unique_ptr<OperationPass<ModuleOp>> createMarkNoAliasArgsPass();

} // namespace triton
} // namespace mlir

#endif // TRITON_TRANSFORM_MARKNOALIASARGS_MARKNOALIASARGS_H
