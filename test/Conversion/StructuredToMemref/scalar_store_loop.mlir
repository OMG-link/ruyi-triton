// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

// Carry an element offset through the loop instead of requiring runtime
// shape metadata for the incoming pointer.
module {
  tt.func @reduce_kernel_2d_0d(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: i32, %arg2: i32, %arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32) {
    %c8_i32 = arith.constant 8 : i32
    %c0_i32 = arith.constant 0 : i32
    %c1_i32 = arith.constant 1 : i32
    %0 = scf.for %arg7 = %c0_i32 to %c8_i32 step %c1_i32 iter_args(%arg8 = %arg0) -> (!tt.ptr<f32>)  : i32 {
      %1 = arith.sitofp %arg7 : i32 to f32
      tt.store %arg8, %1 : !tt.ptr<f32>
      %2 = tt.addptr %arg8, %c1_i32 : !tt.ptr<f32>, i32
      scf.yield %2 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK-LABEL: func.func @reduce_kernel_2d_0d(
// CHECK-SAME: [[BASE:%[a-zA-Z0-9_.$-]+]]: memref<f32> {tt.divisibility = 16 : i32}, {{.*}}) attributes {llvm.bareptr}
// CHECK-DAG: [[ONE:%[a-zA-Z0-9_.$-]+]] = arith.constant 1 : i32
// CHECK-DAG: [[EIGHT:%[a-zA-Z0-9_.$-]+]] = arith.constant 8 : i32
// CHECK-DAG: [[ZERO:%[a-zA-Z0-9_.$-]+]] = arith.constant 0 : i32
// CHECK: [[RESULT:%[a-zA-Z0-9_.$-]+]] = scf.for [[IV:%[a-zA-Z0-9_.$-]+]] = [[ZERO]] to [[EIGHT]] step [[ONE]] iter_args([[OFFSET:%[a-zA-Z0-9_.$-]+]] = [[ZERO]]) -> (i32) : i32 {
// CHECK: [[VALUE:%[a-zA-Z0-9_.$-]+]] = arith.sitofp [[IV]] : i32 to f32
// CHECK: [[INDEX:%[a-zA-Z0-9_.$-]+]] = arith.index_cast [[OFFSET]] : i32 to index
// CHECK: [[ELEMENT:%[a-zA-Z0-9_.$-]+]] = memref.reinterpret_cast [[BASE]] to offset: {{\[}}[[INDEX]]], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK: affine.store [[VALUE]], [[ELEMENT]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK: [[NEXT:%[a-zA-Z0-9_.$-]+]] = arith.addi [[OFFSET]], [[ONE]] : i32
// CHECK: scf.yield [[NEXT]] : i32
// CHECK: return
