// RUN: triton-shared-opt --decompose-high-level-ops %s | FileCheck %s

// Verify that zero accumulators (both tt.splat and dense<0.0> patterns) are
// detected and do NOT generate an extra arith.addf after the reduce.
// Uses concrete shapes M=4, K=8, N=16.

module {
  // Test 1: Zero accumulator via arith.constant dense<0.0> (isSplat pattern)
  tt.func @dot_f32_dense_zero_acc(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %zero_acc = arith.constant dense<0.000000e+00> : tensor<4x16xf32>
    %result = tt.dot %a, %b, %zero_acc : tensor<4x8xf32> * tensor<8x16xf32> -> tensor<4x16xf32>
    tt.return %result : tensor<4x16xf32>
  }

  // Test 2: Zero accumulator via tt.splat (already tested in basic_f32.mlir,
  // included here for completeness alongside the dense pattern)
  tt.func @dot_f32_splat_zero_acc(%a: tensor<4x8xf32>, %b: tensor<8x16xf32>) -> tensor<4x16xf32> {
    %cst = arith.constant 0.000000e+00 : f32
    %zero_acc = tt.splat %cst : f32 -> tensor<4x16xf32>
    %result = tt.dot %a, %b, %zero_acc : tensor<4x8xf32> * tensor<8x16xf32> -> tensor<4x16xf32>
    tt.return %result : tensor<4x16xf32>
  }
}

// --- Test 1: dense<0.0> pattern ---
// CHECK-LABEL: tt.func @dot_f32_dense_zero_acc
// CHECK-SAME:  (%[[A:.*]]: tensor<4x8xf32>, %[[B:.*]]: tensor<8x16xf32>)

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

// No arith.addf for accumulator addition (zero accumulator optimization)
// CHECK-NOT: arith.addf{{.*}}tensor<4x16xf32>

// CHECK: tt.return %[[REDUCE]] : tensor<4x16xf32>

// --- Test 2: tt.splat(0) pattern ---
// CHECK-LABEL: tt.func @dot_f32_splat_zero_acc
// CHECK-SAME:  (%[[A2:.*]]: tensor<4x8xf32>, %[[B2:.*]]: tensor<8x16xf32>)

// Verify decomposition sequence
// CHECK: tt.trans %[[B2]] {order = array<i32: 1, 0>} : tensor<8x16xf32> -> tensor<16x8xf32>
// CHECK: tt.expand_dims %[[A2]] {axis = 1 : i32} : tensor<4x8xf32> -> tensor<4x1x8xf32>
// CHECK: tt.expand_dims {{.*}} {axis = 0 : i32} : tensor<16x8xf32> -> tensor<1x16x8xf32>
// CHECK: tt.broadcast {{.*}} : tensor<4x1x8xf32> -> tensor<4x16x8xf32>
// CHECK: tt.broadcast {{.*}} : tensor<1x16x8xf32> -> tensor<4x16x8xf32>
// CHECK: arith.mulf {{.*}} : tensor<4x16x8xf32>
// CHECK: %[[REDUCE2:.*]] = "tt.reduce"({{.*}}) <{axis = 2 : i32}>
// CHECK:   arith.addf
// CHECK:   tt.reduce.return
// CHECK: }) : (tensor<4x16x8xf32>) -> tensor<4x16xf32>

// No arith.addf for accumulator addition (zero accumulator optimization)
// CHECK-NOT: arith.addf{{.*}}tensor<4x16xf32>

// CHECK: tt.return %[[REDUCE2]] : tensor<4x16xf32>
