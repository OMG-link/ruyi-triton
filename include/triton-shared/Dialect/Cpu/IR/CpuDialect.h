//===- CpuDialect.h - CPU scheduling IR ------------------------*- C++ -*-===//
// Licensed under the Apache License, Version 2.0.
// The Buddy and triton-shared CPU dialect definitions are independent copies
// and must remain identical so both parsers agree on cpu.sched_barrier.
#ifndef TRITON_SHARED_CPU_DIALECT_H
#define TRITON_SHARED_CPU_DIALECT_H
#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/StringRef.h"
namespace mlir::triton_shared::cpu {
struct SchedBarrierResource
    : public SideEffects::Resource::Base<SchedBarrierResource> {
  llvm::StringRef getName() const final { return "SchedBarrierResource"; }
};
} // namespace mlir::triton_shared::cpu
#include "triton-shared/Dialect/Cpu/IR/CpuDialect.h.inc"
#define GET_OP_CLASSES
#include "triton-shared/Dialect/Cpu/IR/Cpu.h.inc"
#endif
