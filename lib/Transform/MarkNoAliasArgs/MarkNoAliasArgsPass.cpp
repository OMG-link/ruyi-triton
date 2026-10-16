// Licensed under the Apache License v2.0.
// SPDX-License-Identifier: Apache-2.0
//===- MarkNoAliasArgsPass.cpp - Mark pointer arguments llvm.noalias -----===//
//
// Sets the `llvm.noalias` unit attribute on every `!tt.ptr` and memref
// argument of every function-like op in the module. The pass encodes the
// user's no-alias guarantee and checks nothing. It is idempotent and leaves
// scalar arguments and existing argument attributes untouched.
//
//===----------------------------------------------------------------------===//

#include "triton-shared/Transform/MarkNoAliasArgs/MarkNoAliasArgs.h"

#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#define DEBUG_TYPE "mark-noalias-args"

using namespace mlir;
using namespace triton;

#define GEN_PASS_CLASSES
#include "triton-shared/Transform/MarkNoAliasArgs/Passes.h.inc"

namespace {

/// Whether an argument of this type refers to memory and gets the mark.
bool isPointerLike(Type type) {
  return isa<triton::PointerType, MemRefType, UnrankedMemRefType>(type);
}

class MarkNoAliasArgsPass : public MarkNoAliasArgsBase<MarkNoAliasArgsPass> {
public:
  void runOnOperation() override {
    // Spelled out so the pass needs no LLVM dialect dependency.
    StringRef attrName = "llvm.noalias";
    UnitAttr unit = UnitAttr::get(&getContext());
    getOperation()->walk([&](FunctionOpInterface func) {
      // Declarations have arguments too, but no body to alias in; marking
      // them is harmless and keeps the rule uniform.
      for (unsigned i = 0, e = func.getNumArguments(); i < e; ++i) {
        if (!isPointerLike(func.getArgumentTypes()[i]))
          continue;
        if (func.getArgAttr(i, attrName))
          continue;
        func.setArgAttr(i, attrName, unit);
      }
    });
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> triton::createMarkNoAliasArgsPass() {
  return std::make_unique<MarkNoAliasArgsPass>();
}
