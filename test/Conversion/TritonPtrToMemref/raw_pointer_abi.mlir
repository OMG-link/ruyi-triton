// RUN: triton-shared-opt %s --triton-ptr-to-memref --reconcile-unrealized-casts | FileCheck %s --check-prefix=MEMREF
// RUN: triton-shared-opt %s --triton-ptr-to-memref --reconcile-unrealized-casts --convert-func-to-llvm --finalize-memref-to-llvm --convert-arith-to-llvm --reconcile-unrealized-casts | FileCheck %s --check-prefix=LLVM

// The external pointer has no shape metadata. The rank-zero carrier supplies
// exactly the descriptor fields that the old launcher used to construct.
func.func @load(%pointer: !tt.ptr<f32>, %index: index) -> f32 {
  %base = builtin.unrealized_conversion_cast %pointer : !tt.ptr<f32> to memref<*xf32>
  %view = memref.reinterpret_cast %base to offset: [0], sizes: [16], strides: [1] : memref<*xf32> to memref<16xf32>
  %value = memref.load %view[%index] : memref<16xf32>
  return %value : f32
}

// MEMREF-LABEL: func.func @load(
// MEMREF-SAME: %[[POINTER:.*]]: memref<f32>, %{{.*}}: index) -> f32 attributes {llvm.bareptr}
// MEMREF: memref.reinterpret_cast %[[POINTER]]
// LLVM-LABEL: llvm.func @load(
// LLVM-SAME: %[[POINTER:.*]]: !llvm.ptr, %{{.*}}: i64) -> f32
// LLVM: %[[EMPTY:.*]] = llvm.mlir.poison : !llvm.struct<(ptr, ptr, i64)>
// LLVM: %[[ALLOCATED:.*]] = llvm.insertvalue %[[POINTER]], %[[EMPTY]][0]
// LLVM: %[[ALIGNED:.*]] = llvm.insertvalue %[[POINTER]], %[[ALLOCATED]][1]
// LLVM: %[[ZERO:.*]] = llvm.mlir.constant(0 : i64) : i64
// LLVM: llvm.insertvalue %[[ZERO]], %[[ALIGNED]][2]

func.func private @consume(!tt.ptr<f32>)
func.func @caller(%pointer: !tt.ptr<f32>) {
  func.call @consume(%pointer) : (!tt.ptr<f32>) -> ()
  return
}

// MEMREF: func.func private @consume(memref<f32>) attributes {llvm.bareptr}
// MEMREF-LABEL: func.func @caller(
// MEMREF-SAME: %[[POINTER:.*]]: memref<f32>) attributes {llvm.bareptr}
// MEMREF: call @consume(%[[POINTER]]) : (memref<f32>) -> ()
// LLVM: llvm.func @consume(!llvm.ptr)
// LLVM-LABEL: llvm.func @caller(%{{.*}}: !llvm.ptr)
// LLVM: llvm.call @consume(%{{.*}}) : (!llvm.ptr) -> ()

func.func @identity(%pointer: !tt.ptr<f32>) -> !tt.ptr<f32> {
  return %pointer : !tt.ptr<f32>
}

// MEMREF-LABEL: func.func @identity(
// MEMREF-SAME: %[[POINTER:.*]]: memref<f32>) -> memref<f32> attributes {llvm.bareptr}
// MEMREF: return %[[POINTER]] : memref<f32>
// LLVM-LABEL: llvm.func @identity(%{{.*}}: !llvm.ptr) -> !llvm.ptr
// LLVM: llvm.return %{{.*}} : !llvm.ptr
