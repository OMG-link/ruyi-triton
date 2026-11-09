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
LogicalResult triton_shared::cpu::ShuffleOp::verify() {
  auto v1 = cast<ShapedType>(getV1().getType());
  auto v2 = cast<ShapedType>(getV2().getType());
  auto result = cast<ShapedType>(getResult().getType());
  bool isVector = isa<VectorType>(v1);
  if (isa<VectorType>(v2) != isVector || isa<VectorType>(result) != isVector)
    return emitOpError(
        "operands and result must be all tensors or all vectors");
  for (ShapedType type : {v1, v2, result})
    if (auto vector = dyn_cast<VectorType>(type); vector && vector.isScalable())
      return emitOpError("scalable vectors are not supported");
  if (v1.getElementType() != v2.getElementType() ||
      result.getElementType() != v1.getElementType())
    return emitOpError("operands and result must have one element type");
  if (v1.getRank() != v2.getRank() || result.getRank() != v1.getRank())
    return emitOpError("operands and result must have one rank");
  if (!v1.hasStaticShape() || !v2.hasStaticShape() || !result.hasStaticShape())
    return emitOpError("shapes must be static");
  if (v1.getShape().drop_front() != v2.getShape().drop_front() ||
      result.getShape().drop_front() != v1.getShape().drop_front())
    return emitOpError("operands and result must agree in every dimension "
                       "but the first");
  ArrayRef<int64_t> mask = getMask();
  if (static_cast<int64_t>(mask.size()) != result.getDimSize(0))
    return emitOpError("result's first dimension must equal the mask size");
  int64_t bound = v1.getDimSize(0) + v2.getDimSize(0);
  for (int64_t index : mask)
    if (index < -1 || index >= bound)
      return emitOpError("mask element ")
             << index << " is out of range [-1, " << bound << ")";
  return success();
}
