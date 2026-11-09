//===- Cpu.td - CPU-specific IR operations --------------*- tablegen -*-===//
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//
//
// The `cpu` dialect carries CPU-specific IR operations. The same dialect is defined in triton-riscv
// (include/triton-shared/Dialect/Cpu/IR/Cpu.td), which emits these ops into
// the Linalg IR buddy-opt reads; the two definitions must stay identical.
//
//===----------------------------------------------------------------------===//

#ifndef CPU_DIALECT
#define CPU_DIALECT

include "mlir/IR/OpBase.td"
include "mlir/Interfaces/InferTypeOpInterface.td"
include "mlir/Interfaces/SideEffectInterfaces.td"

def SchedBarrierResource : Resource<"::mlir::triton_shared::cpu::SchedBarrierResource">;

def Cpu_Dialect : Dialect {
  let name = "cpu";
  let cppNamespace = "::mlir::triton_shared::cpu";
  let summary = "CPU-specific operations";
}

class Cpu_Op<string mnemonic, list<Trait> traits = []> :
    Op<Cpu_Dialect, mnemonic, traits>;

def Cpu_SchedBarrierOp : Cpu_Op<"sched_barrier", [
    DeclareOpInterfaceMethods<InferTypeOpInterface>,
    DeclareOpInterfaceMethods<MemoryEffectsOpInterface>]> {
  let summary = "Instruction scheduling barrier on a set of values";
  let description = [{
    Returns its operands unchanged and orders the code around them: the
    computation that produces the operands stays before the barrier, and the
    code that uses the results stays after it. The `memory` attribute optionally
    adds a memory clobber.
    Code that uses the operands instead of the results is not ordered.

    It lowers to an empty inline asm that takes every operand in a vector
    register (or register group) and returns it there, the form a
    hand-written kernel uses (`asm volatile("" : "+vr"(x))`). The `memory`
    attribute adds the `"memory"` clobber.
    Each operand must fill exactly one register or register group of the
    target; lowering reports an error otherwise.
  }];
  let arguments = (ins Variadic<AnyType>:$inputs, OptionalAttr<UnitAttr>:$memory);
  let results = (outs Variadic<AnyType>:$results);
  let assemblyFormat = "($inputs^ `:` type($inputs))? prop-dict attr-dict";
  let hasVerifier = 1;
}

def Cpu_ShuffleOp : Cpu_Op<"shuffle", [Pure]> {
  let summary = "Constant shuffle of two tensors or two vectors";
  let description = [{
    `vector.shuffle` on ranked tensors as well as vectors. `v1`, `v2` and the
    result are all tensors or all vectors, with one element type, one rank and
    the same sizes in every dimension but the first. The result's first
    dimension has one entry per `mask` element. Mask element `i` selects slice
    `i` of `v1` when `i < v1.shape[0]`, slice `i - v1.shape[0]` of `v2`
    otherwise, and an undefined (poison) slice when it is -1.

    The tensor form is what triton-shared emits; tile tensor-to-vector turns it
    into `vector.shuffle` with the same mask.
  }];
  let arguments = (ins AnyTypeOf<[AnyRankedTensor, AnyVectorOfNonZeroRank]>:$v1,
                       AnyTypeOf<[AnyRankedTensor, AnyVectorOfNonZeroRank]>:$v2,
                       DenseI64ArrayAttr:$mask);
  let results = (outs AnyTypeOf<[AnyRankedTensor, AnyVectorOfNonZeroRank]>:$result);
  let assemblyFormat = [{
    $v1 `,` $v2 $mask attr-dict `:` type($v1) `,` type($v2) `->` type($result)
  }];
  let hasVerifier = 1;
}

#endif // CPU_DIALECT
