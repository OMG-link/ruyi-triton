// RUN: triton-shared-opt --triton-to-unstructured %s | FileCheck %s

// tts.make_tptr whose offsets are all static (no offset operand) and whose
// base comes from tt.int_to_ptr used to crash (out of bounds access to the
// offset operands).

module {
  // Base is the int_to_ptr root itself: nothing to fold.
  tt.func @root_base(%a: i64) -> tensor<16xf32> {
    %p = tt.int_to_ptr %a : i64 -> !tt.ptr<f32>
    %t = tts.make_tptr %p to sizes: [16], strides: [1], offsets: [0], shape: [0], order: [] : <f32> to tensor<16x!tt.ptr<f32>>
    %v = tt.load %t : tensor<16x!tt.ptr<f32>>
    tt.return %v : tensor<16xf32>
  }

  // Base is offset from the root: the offset is folded into the first
  // (previously static) offset.
  tt.func @offset_base(%a: i64, %k: i32) -> tensor<4x8xf32> {
    %p = tt.int_to_ptr %a : i64 -> !tt.ptr<f32>
    %q = tt.addptr %p, %k : !tt.ptr<f32>, i32
    %t = tts.make_tptr %q to sizes: [4, 8], strides: [8, 1], offsets: [0, 16], shape: [0, 0], order: [] : <f32> to tensor<4x8x!tt.ptr<f32>>
    %v = tt.load %t : tensor<4x8x!tt.ptr<f32>>
    tt.return %v : tensor<4x8xf32>
  }
}

// CHECK-LABEL: tt.func @root_base
// CHECK:         [[P:%.+]] = tt.int_to_ptr
// CHECK:         tts.make_tptr [[P]] to sizes: [16], strides: [1], offsets: [0], shape: [0], order: []

// CHECK-LABEL: tt.func @offset_base
// CHECK-SAME:    ([[A:%.+]]: i64, [[K:%.+]]: i32)
// CHECK:         [[P:%.+]] = tt.int_to_ptr [[A]]
// CHECK:         [[FIRST:%.+]] = arith.index_cast [[K]] : i32 to index
// CHECK:         tts.make_tptr [[P]] to sizes: [4, 8], strides: [8, 1], offsets: {{\[}}[[FIRST]], 16], shape: [0, 0], order: []
// CHECK-NOT:     tt.addptr
