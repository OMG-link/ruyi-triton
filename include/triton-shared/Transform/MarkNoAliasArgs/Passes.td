// Licensed under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
#ifndef MARK_NOALIAS_ARGS_TRANSFORM_PASSES
#define MARK_NOALIAS_ARGS_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def MarkNoAliasArgs : Pass<"mark-noalias-args", "mlir::ModuleOp"> {
  let summary = "Mark every pointer and memref function argument llvm.noalias";
  let description = [{
    Sets the `llvm.noalias` unit attribute on each argument of type
    `!tt.ptr` or memref of every function-like op in the module. Scalar
    arguments and existing argument attributes are left untouched, and the
    pass is idempotent. The pass trusts the user's guarantee that no two
    pointer arguments overlap; it checks nothing.
  }];
  let constructor = "triton::createMarkNoAliasArgsPass()";
}

#endif
