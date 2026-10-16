// Licensed under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
#ifndef MARK_NOALIAS_ARGS_TRANSFORM_PASSES_H
#define MARK_NOALIAS_ARGS_TRANSFORM_PASSES_H

#include "triton-shared/Transform/MarkNoAliasArgs/MarkNoAliasArgs.h"

namespace mlir {
namespace triton {

#define GEN_PASS_REGISTRATION
#include "triton-shared/Transform/MarkNoAliasArgs/Passes.h.inc"

} // namespace triton
} // namespace mlir

#endif
