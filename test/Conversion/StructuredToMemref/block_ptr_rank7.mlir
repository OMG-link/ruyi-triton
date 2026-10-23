// RUN: triton-shared-opt --structured-to-memref %s | FileCheck %s
//
// A rank-7 structured pointer. Its seven strides no longer fit the inline
// storage of the SmallVector holding them, which exposed a use-after-free in
// getMixedStridesForMemref: it iterated a reversed zip over a temporary.

module {
  tt.func public @rank7_ptr(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>) {
    %0 = tts.make_tptr %arg0 to sizes: [2, 2, 2, 2, 2, 4, 8], strides: [512, 256, 128, 64, 32, 8, 1], offsets: [0, 0, 0, 0, 0, 0, 0], shape: [0, 0, 0, 0, 0, 0, 0], order: [] : <i8> to tensor<2x2x2x2x2x4x8x!tt.ptr<i8>>
    %1 = tts.make_tptr %arg1 to sizes: [2, 2, 2, 2, 2, 4, 8], strides: [512, 256, 128, 64, 32, 8, 1], offsets: [0, 0, 0, 0, 0, 0, 0], shape: [0, 0, 0, 0, 0, 0, 0], order: [] : <i8> to tensor<2x2x2x2x2x4x8x!tt.ptr<i8>>
    %2 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<2x2x2x2x2x4x8x!tt.ptr<i8>>) -> tensor<2x2x2x2x2x4x8xi8>
    "tts.store"(%1, %2) <{static_mask_dims = array<i64>}> : (tensor<2x2x2x2x2x4x8x!tt.ptr<i8>>, tensor<2x2x2x2x2x4x8xi8>) -> ()
    tt.return
  }
}

// CHECK-LABEL: tt.func public @rank7_ptr
// CHECK: memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [0], sizes: [2, 2, 2, 2, 2, 4, 8], strides: [512, 256, 128, 64, 32, 8, 1]
// CHECK: memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [0], sizes: [2, 2, 2, 2, 2, 4, 8], strides: [512, 256, 128, 64, 32, 8, 1]
// CHECK: bufferization.materialize_in_destination
