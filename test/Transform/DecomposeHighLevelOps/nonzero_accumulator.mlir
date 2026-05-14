// RUN: triton-shared-opt --decompose-high-level-ops %s | FileCheck %s

// Verify that non-zero accumulators generate arith.addf (for floats) or
// arith.addi (for integers) after the reduce operation.
// Uses concrete shapes M=4, K=8, N=16.

module {
  // Test 1: Non-zero f32 accumulator (passed as function argument)
  tt.func @dot_f32_nonzero_acc(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>, %c: tensor<4x16xf32>) -> tensor<4x16xf32> {
    %result = tt.dot %a, %b, %c : tensor<4x8xf32> * tensor<8x16xf32> -> tensor<4x16xf32>
    tt.return %result : tensor<4x16xf32>
  }

  // Test 2: Non-zero i32 accumulator (passed as function argument)
  tt.func @dot_i32_nonzero_acc(%a: tensor<4x8xi32>, %b: tensor<8x16xi32>, %c: tensor<4x16xi32>) -> tensor<4x16xi32> {
    %result = tt.dot %a, %b, %c : tensor<4x8xi32> * tensor<8x16xi32> -> tensor<4x16xi32>
    tt.return %result : tensor<4x16xi32>
  }
}

// --- Test 1: f32 non-zero accumulator ---
// CHECK-LABEL: tt.func @dot_f32_nonzero_acc
// CHECK-SAME:  (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x16xf32>, %[[C:.*]]: tensor<4x16xf32>)

// Verify decomposition sequence
// CHECK: tt.trans %[[B]] {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
// CHECK: tt.expand_dims %[[A]] {axis = 1 : i32} : tensor<4x8xf32> -> tensor<4x1x8xf32>
// CHECK: tt.expand_dims {{.*}} {axis = 0 : i32} : tensor<16x8xf32> -> tensor<1x16x8xf32>
// CHECK: tt.broadcast {{.*}} : tensor<4x1x8xf32> -> tensor<4x16x8xf32>
// CHECK: tt.broadcast {{.*}} : tensor<1x16x8xf32> -> tensor<4x16x8xf32>
// CHECK: arith.mulf {{.*}} : tensor<4x16x8xf32>
// CHECK: %[[REDUCE:.*]] = "tt.reduce"({{.*}}) <{axis = 2 : i32}>
// CHECK:   arith.addf
// CHECK:   tt.reduce.return
// CHECK: }) : (tensor<4x16x8xf32>) -> tensor<4x16xf32>

// Verify arith.addf for accumulator addition (non-zero accumulator)
// CHECK: %[[RESULT:.*]] = arith.addf %[[REDUCE]], %[[C]] : tensor<4x16xf32>
// CHECK: tt.return %[[RESULT]] : tensor<4x16xf32>

// --- Test 2: i32 non-zero accumulator ---
// CHECK-LABEL: tt.func @dot_i32_nonzero_acc
// CHECK-SAME:  (%[[A2:.*]]: tensor<4x8xi32>, %[[B2:.*]]: tensor<8x16xi32>, %[[C2:.*]]: tensor<4x16xi32>)

// Verify decomposition sequence with integer ops
// CHECK: tt.trans %[[B2]] {order = array<i32: 1, 0>} : tensor<8x16xi32> -> tensor<16x8xi32>
// CHECK: tt.expand_dims %[[A2]] {axis = 1 : i32} : tensor<4x8xi32> -> tensor<4x1x8xi32>
// CHECK: tt.expand_dims {{.*}} {axis = 0 : i32} : tensor<16x8xi32> -> tensor<1x16x8xi32>
// CHECK: tt.broadcast {{.*}} : tensor<4x1x8xi32> -> tensor<4x16x8xi32>
// CHECK: tt.broadcast {{.*}} : tensor<1x16x8xi32> -> tensor<4x16x8xi32>
// CHECK: arith.muli {{.*}} : tensor<4x16x8xi32>
// CHECK: %[[REDUCE2:.*]] = "tt.reduce"({{.*}}) <{axis = 2 : i32}>
// CHECK:   arith.addi
// CHECK:   tt.reduce.return
// CHECK: }) : (tensor<4x16x8xi32>) -> tensor<4x16xi32>

// Verify arith.addi for accumulator addition (non-zero integer accumulator)
// CHECK: %[[RESULT2:.*]] = arith.addi %[[REDUCE2]], %[[C2]] : tensor<4x16xi32>
// CHECK: tt.return %[[RESULT2]] : tensor<4x16xi32>
