// RUN: triton-shared-opt --decompose-high-level-ops %s | FileCheck %s

// Verify that a f16×f16 DotOp with f32 output is correctly decomposed
// with arith.extf inserted to widen f16 inputs to f32 before multiplication.
// Uses concrete shapes M=4, K=8, N=16.

module {
  tt.func @dot_f16_to_f32(%a: tensor<4x8xf16>, %b: tensor<8x16xf16>) -> tensor<4x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %zero_acc = tt.splat %cst : f32 -> tensor<4x16xf32>
    %result = tt.dot %a, %b, %zero_acc : tensor<4x8xf16> * tensor<8x16xf16> -> tensor<4x16xf32>
    tt.return %result : tensor<4x16xf32>
  }
}

// CHECK-LABEL: tt.func @dot_f16_to_f32
// CHECK-SAME:  (%[[A:.*]]: tensor<4x8xf16>, %[[B:.*]]: tensor<8x16xf16>)

// Step 1: Transpose b from (8, 16) to (16, 8)
// CHECK: %[[B_T:.*]] = tt.trans %[[B]] {order = array<i32: 1, 0>} : tensor<8x16xf16> -> tensor<16x8xf16>

// Step 2: Type conversion - extend f16 inputs to f32
// CHECK: %[[A_EXT:.*]] = arith.extf %[[A]] : tensor<4x8xf16> to tensor<4x8xf32>
// CHECK: %[[B_EXT:.*]] = arith.extf %[[B_T]] : tensor<16x8xf16> to tensor<16x8xf32>

// Step 3: Expand dims - a from (4, 8) to (4, 1, 8)
// CHECK: %[[A_EXP:.*]] = tt.expand_dims %[[A_EXT]] {axis = 1 : i32} : tensor<4x8xf32> -> tensor<4x1x8xf32>

// Step 4: Expand dims - b_t from (16, 8) to (1, 16, 8)
// CHECK: %[[B_EXP:.*]] = tt.expand_dims %[[B_EXT]] {axis = 0 : i32} : tensor<16x8xf32> -> tensor<1x16x8xf32>

// Step 5: Broadcast a_exp to (4, 16, 8)
// CHECK: %[[A_BC:.*]] = tt.broadcast %[[A_EXP]] : tensor<4x1x8xf32> -> tensor<4x16x8xf32>

// Step 6: Broadcast b_exp to (4, 16, 8)
// CHECK: %[[B_BC:.*]] = tt.broadcast %[[B_EXP]] : tensor<1x16x8xf32> -> tensor<4x16x8xf32>

// Step 7: Elementwise multiplication (float)
// CHECK: %[[PROD:.*]] = arith.mulf %[[A_BC]], %[[B_BC]] : tensor<4x16x8xf32>

// Step 8: Reduce along axis 2 with addf combiner
// CHECK: %[[REDUCE:.*]] = "tt.reduce"(%[[PROD]]) <{axis = 2 : i32}>
// CHECK:   ^bb0(%[[ARG0:.*]]: f32, %[[ARG1:.*]]: f32):
// CHECK:     %[[ADD:.*]] = arith.addf %[[ARG0]], %[[ARG1]] : f32
// CHECK:     tt.reduce.return %[[ADD]] : f32
// CHECK: }) : (tensor<4x16x8xf32>) -> tensor<4x16xf32>

// No arith.addf for accumulator (zero accumulator optimization)
// CHECK-NOT: arith.addf{{.*}}tensor<4x16xf32>

// CHECK: tt.return %[[REDUCE]] : tensor<4x16xf32>
