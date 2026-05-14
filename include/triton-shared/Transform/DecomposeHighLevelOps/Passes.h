// Licensed under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0

#ifndef DECOMPOSE_HIGH_LEVEL_OPS_TRANSFORM_PASSES_H
#define DECOMPOSE_HIGH_LEVEL_OPS_TRANSFORM_PASSES_H

#include "triton-shared/Transform/DecomposeHighLevelOps/DecomposeHighLevelOps.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Transform/DecomposeHighLevelOps/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
