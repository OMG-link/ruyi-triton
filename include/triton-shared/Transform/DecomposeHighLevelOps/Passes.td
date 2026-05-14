// Licensed under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0

#ifndef DECOMPOSE_HIGH_LEVEL_OPS_TRANSFORM_PASSES
#define DECOMPOSE_HIGH_LEVEL_OPS_TRANSFORM_PASSES

include "mlir/Pass/PassBase.td"

def DecomposeHighLevelOps : Pass<"decompose-high-level-ops", "mlir::ModuleOp"> {
  let summary = "Decompose high-level Triton ops (tt.dot) into basic ops";
  let constructor = "triton::createDecomposeHighLevelOpsPass()";
  let dependentDialects = [
    "triton::TritonDialect",
    "arith::ArithDialect"
  ];
}

#endif
