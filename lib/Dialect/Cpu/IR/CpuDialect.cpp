//===- CpuDialect.cpp - CPU scheduling IR ----------------------*- C++ -*-===//
// Licensed under the Apache License, Version 2.0.
#include "triton-shared/Dialect/Cpu/IR/CpuDialect.h"
using namespace mlir;
#include "triton-shared/Dialect/Cpu/IR/CpuDialect.cpp.inc"
#define GET_OP_CLASSES
#include "triton-shared/Dialect/Cpu/IR/Cpu.cpp.inc"
void triton_shared::cpu::CpuDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "triton-shared/Dialect/Cpu/IR/Cpu.cpp.inc"
      >();
}
LogicalResult triton_shared::cpu::SchedBarrierOp::inferReturnTypes(
    MLIRContext *, std::optional<Location>, ValueRange operands, DictionaryAttr,
    PropertyRef, RegionRange, SmallVectorImpl<Type> &inferredReturnTypes) {
  llvm::append_range(inferredReturnTypes, operands.getTypes());
  return success();
}
LogicalResult triton_shared::cpu::SchedBarrierOp::verify() {
  if (getInputs().empty())
    return emitOpError("needs at least one operand");
  return success();
}
void triton_shared::cpu::SchedBarrierOp::getEffects(
    SmallVectorImpl<MemoryEffects::EffectInstance> &effects) {
  effects.emplace_back(MemoryEffects::Write::get(),
                       SchedBarrierResource::get());
  if (getMemoryAttr()) {
    effects.emplace_back(MemoryEffects::Read::get(),
                         SideEffects::DefaultResource::get());
    effects.emplace_back(MemoryEffects::Write::get(),
                         SideEffects::DefaultResource::get());
  }
}
