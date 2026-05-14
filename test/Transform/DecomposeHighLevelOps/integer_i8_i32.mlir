// RUN: triton-shared-opt --decompose-high-level-ops %s | FileCheck %s

// Verify that an i8×i8 DotOp with i32 output is correctly decomposed
// with arith.extsi inserted to widen i8 inputs to i32, and uses
// arith.muli + arith.addi for integer arithmetic.
// Uses concrete shapes M=4, K=8, N=16.

module {
  tt.func @dot_i8_to_i32(%a: tensor<4x8xi8>, %b: tensor<8x16xi8>) -> tensor<4x16xi32> {
    %cst = arith.constant 0 : i32
    %zero_acc = tt.splat %cst : i32 -> tensor<4x16xi32>
    %result = tt.dot %a, %b, %zero_acc : tensor<4x8xi8> * tensor<8x16xi8> -> tensor<4x16xi32>
    tt.return %result : tensor<4x16xi32>
  }
}

// CHECK-LABEL: tt.func @dot_i8_to_i32
// CHECK-SAME:  (%[[A:.*]]: tensor<4x8xi8>, %[[B:.*]]: tensor<8x16xi8>)

// Step 1: Transpose b from (8, 16) to (16, 8)
// CHECK: %[[B_T:.*]] = tt.trans %[[B]] {order = array<i32: 1, 0>} : tensor<8x16xi8> -> tensor<16x8xi8>

// Step 2: Type conversion - sign-extend i8 inputs to i32
// CHECK: %[[A_EXT:.*]] = arith.extsi %[[A]] : tensor<4x8xi8> to tensor<4x8xi32>
// CHECK: %[[B_EXT:.*]] = arith.extsi %[[B_T]] : tensor<16x8xi8> to tensor<16x8xi32>

// Step 3: Expand dims - a from (4, 8) to (4, 1, 8)
// CHECK: %[[A_EXP:.*]] = tt.expand_dims %[[A_EXT]] {axis = 1 : i32} : tensor<4x8xi32> -> tensor<4x1x8xi32>

// Step 4: Expand dims - b_t from (16, 8) to (1, 16, 8)
// CHECK: %[[B_EXP:.*]] = tt.expand_dims %[[B_EXT]] {axis = 0 : i32} : tensor<16x8xi32> -> tensor<1x16x8xi32>

// Step 5: Broadcast a_exp to (4, 16, 8)
// CHECK: %[[A_BC:.*]] = tt.broadcast %[[A_EXP]] : tensor<4x1x8xi32> -> tensor<4x16x8xi32>

// Step 6: Broadcast b_exp to (4, 16, 8)
// CHECK: %[[B_BC:.*]] = tt.broadcast %[[B_EXP]] : tensor<1x16x8xi32> -> tensor<4x16x8xi32>

// Step 7: Elementwise multiplication (integer)
// CHECK: %[[PROD:.*]] = arith.muli %[[A_BC]], %[[B_BC]] : tensor<4x16x8xi32>

// Step 8: Reduce along axis 2 with addi combiner
// CHECK: %[[REDUCE:.*]] = "tt.reduce"(%[[PROD]]) <{axis = 2 : i32}>
// CHECK:   ^bb0(%[[ARG0:.*]]: i32, %[[ARG1:.*]]: i32):
// CHECK:     %[[ADD:.*]] = arith.addi %[[ARG0]], %[[ARG1]] : i32
// CHECK:     tt.reduce.return %[[ADD]] : i32
// CHECK: }) : (tensor<4x16x8xi32>) -> tensor<4x16xi32>

// No arith.addi for accumulator (zero accumulator optimization)
// CHECK-NOT: arith.addi{{.*}}tensor<4x16xi32>

// CHECK: tt.return %[[REDUCE]] : tensor<4x16xi32>
