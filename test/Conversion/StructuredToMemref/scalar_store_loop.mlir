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

// CHECK lines should be minimized and named to reflect the test’s intent.
// CHECK-LABEL:   func.func @reduce_kernel_2d_0d(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {llvm.bareptr} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 8 : i32
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 0 : index
// CHECK:           %[[FOR_0:.*]] = scf.for %[[VAL_0:.*]] = %[[CONSTANT_2]] to %[[CONSTANT_3]] step %[[CONSTANT_1]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_4]]) -> (index)  : i32 {
// CHECK:             %[[SITOFP_0:.*]] = arith.sitofp %[[VAL_0]] : i32 to f32
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: {{\[}}%[[VAL_1]]], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             affine.store %[[SITOFP_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_0]] : index
// CHECK:             scf.yield %[[ADDI_0]] : index
// CHECK:           }
// CHECK:           return
// CHECK:         }
