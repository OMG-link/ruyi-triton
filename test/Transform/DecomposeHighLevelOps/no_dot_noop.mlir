// RUN: triton-shared-opt --decompose-high-level-ops %s | FileCheck %s

// Verify that when no tt.dot operations exist in the module, the pass does
// not modify the IR. All original operations should remain unchanged.

module {
  tt.func @no_dot_operations(%arg0: f32, %arg1: tensor<4x8xf32>, %arg2: tensor<4x8xf32>) -> tensor<4x8xf32> {
    // Some triton ops that are NOT tt.dot
    %splat = tt.splat %arg0 : f32 -> tensor<4x8xf32>
    %add = arith.addf %arg1, %arg2 : tensor<4x8xf32>
    %mul = arith.mulf %add, %splat : tensor<4x8xf32>
    tt.return %mul : tensor<4x8xf32>
  }
}

// CHECK-LABEL: tt.func @no_dot_operations
// CHECK-SAME:  (%[[ARG0:.*]]: f32, %[[ARG1:.*]]: tensor<4x8xf32>, %[[ARG2:.*]]: tensor<4x8xf32>)

// Verify all original ops are preserved unchanged
// CHECK: %[[SPLAT:.*]] = tt.splat %[[ARG0]] : f32 -> tensor<4x8xf32>
// CHECK: %[[ADD:.*]] = arith.addf %[[ARG1]], %[[ARG2]] : tensor<4x8xf32>
// CHECK: %[[MUL:.*]] = arith.mulf %[[ADD]], %[[SPLAT]] : tensor<4x8xf32>
// CHECK: tt.return %[[MUL]] : tensor<4x8xf32>

// Verify no decomposition ops were introduced
// CHECK-NOT: tt.trans
// CHECK-NOT: tt.expand_dims
// CHECK-NOT: tt.broadcast
// CHECK-NOT: tt.reduce
