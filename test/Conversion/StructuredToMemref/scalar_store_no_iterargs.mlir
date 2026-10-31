// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func @reduce_kernel_2d_0d1d2de3de(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32}, %arg2: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg3: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32}, %arg4: i32, %arg5: i32, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32) {
    %c1_i32 = arith.constant 1 : i32
    %c5_i32 = arith.constant 5 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.addptr %arg1, %arg7 : !tt.ptr<f32>, i32
    %1 = arith.sitofp %arg7 : i32 to f32
    scf.for %arg10 = %c0_i32 to %c5_i32 step %c1_i32  : i32 {
      %2 = tt.addptr %0, %arg10 : !tt.ptr<f32>, i32
      tt.store %2, %1 : !tt.ptr<f32>
    }
    tt.return
  }
}

// CHECK lines should be minimized and named to reflect the test’s intent.
// CHECK-LABEL:   func.func @reduce_kernel_2d_0d1d2de3de(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<f32> {tt.divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32 {tt.divisibility = 16 : i32, tt.max_divisibility = 16 : i32},
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG9:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG10:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG11:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG12:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG13:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG14:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG15:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {llvm.bareptr} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 5 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 1 : i32
// CHECK:           %[[INDEX_CAST_0:.*]] = arith.index_cast %[[ARG7]] : i32 to index
// CHECK:           %[[SITOFP_0:.*]] = arith.sitofp %[[ARG7]] : i32 to f32
// CHECK:           scf.for %[[VAL_0:.*]] = %[[CONSTANT_0]] to %[[CONSTANT_1]] step %[[CONSTANT_2]]  : i32 {
// CHECK:             %[[INDEX_CAST_1:.*]] = arith.index_cast %[[VAL_0]] : i32 to index
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[INDEX_CAST_0]], %[[INDEX_CAST_1]] : index
// CHECK:             %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: {{\[}}%[[ADDI_0]]], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1], offset: ?>>
// CHECK:             affine.store %[[SITOFP_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1], offset: ?>>
// CHECK:           }
// CHECK:           return
// CHECK:         }
