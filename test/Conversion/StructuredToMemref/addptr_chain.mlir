// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @addptr(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %c10_i32 = arith.constant 10 : i32
    %c2_i32 = arith.constant 2 : i32
    %0 = tt.addptr %arg0, %c1_i32 : !tt.ptr<f32>, i32
    %1 = tt.addptr %arg1, %c1_i32 : !tt.ptr<f32>, i32
    scf.for %arg2 = %c0_i32 to %c10_i32 step %c2_i32  : i32 {
      %2 = tt.addptr %0, %arg2 : !tt.ptr<f32>, i32
      %3 = tt.addptr %2, %c1_i32 : !tt.ptr<f32>, i32
      %4 = tt.addptr %1, %arg2 : !tt.ptr<f32>, i32
      %5 = tt.addptr %4, %c1_i32 : !tt.ptr<f32>, i32
      %6 = tt.load %2 : !tt.ptr<f32>
      %7 = tt.load %3 : !tt.ptr<f32>
      tt.store %4, %6 : !tt.ptr<f32>
      tt.store %5, %7 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK lines should be minimized and named to reflect the test’s intent.
// CHECK-LABEL:   func.func @addptr(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<f32>,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<f32>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {llvm.bareptr} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 10 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 2 : index
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_2]] step %[[CONSTANT_3]]  : i32 {
// CHECK:             %[[INDEX_CAST_0:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_0]] : index
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[INDEX_CAST_0]], %[[CONSTANT_4]] : index
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: {{\[}}%[[ADDI_0]]], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[LOAD_0:.*]] = affine.load %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: {{\[}}%[[ADDI_1]]], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[LOAD_1:.*]] = affine.load %[[REINTERPRET_CAST_1]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[REINTERPRET_CAST_2:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: {{\[}}%[[ADDI_0]]], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             affine.store %[[LOAD_0]], %[[REINTERPRET_CAST_2]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[REINTERPRET_CAST_3:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: {{\[}}%[[ADDI_1]]], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             affine.store %[[LOAD_1]], %[[REINTERPRET_CAST_3]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
