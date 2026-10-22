// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
//
// A rank-7 block pointer. Its seven strides no longer fit the inline storage
// of the SmallVector holding them, which exposed a use-after-free in
// getMixedStridesForMemref: it iterated a reversed zip over a temporary.

module {
  tt.func public @rank7_block_ptr(%arg0: !tt.ptr<i8>, %arg1: !tt.ptr<i8>) {
    %c0_i32 = arith.constant 0 : i32
    %c1_i64 = arith.constant 1 : i64
    %c2_i64 = arith.constant 2 : i64
    %c4_i64 = arith.constant 4 : i64
    %c8_i64 = arith.constant 8 : i64
    %c32_i64 = arith.constant 32 : i64
    %c64_i64 = arith.constant 64 : i64
    %c128_i64 = arith.constant 128 : i64
    %c256_i64 = arith.constant 256 : i64
    %c512_i64 = arith.constant 512 : i64
    %0 = tt.make_tensor_ptr %arg0, [%c2_i64, %c2_i64, %c2_i64, %c2_i64, %c2_i64, %c4_i64, %c8_i64], [%c512_i64, %c256_i64, %c128_i64, %c64_i64, %c32_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] {order = array<i32: 6, 5, 4, 3, 2, 1, 0>} : <tensor<2x2x2x2x2x4x8xi8>>
    %1 = tt.make_tensor_ptr %arg1, [%c2_i64, %c2_i64, %c2_i64, %c2_i64, %c2_i64, %c4_i64, %c8_i64], [%c512_i64, %c256_i64, %c128_i64, %c64_i64, %c32_i64, %c8_i64, %c1_i64], [%c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32, %c0_i32] {order = array<i32: 6, 5, 4, 3, 2, 1, 0>} : <tensor<2x2x2x2x2x4x8xi8>>
    %2 = tt.load %0 : !tt.ptr<tensor<2x2x2x2x2x4x8xi8>>
    tt.store %1, %2 : !tt.ptr<tensor<2x2x2x2x2x4x8xi8>>
    tt.return
  }
}

// CHECK-LABEL: func.func @rank7_block_ptr
// CHECK: memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [2, 2, 2, 2, 2, 4, 8], strides: [512, 256, 128, 64, 32, 8, 1]
// CHECK: memref.reinterpret_cast %{{[a-z0-9_]+}} to offset: [%{{[a-z0-9_]+}}], sizes: [2, 2, 2, 2, 2, 4, 8], strides: [512, 256, 128, 64, 32, 8, 1]
// CHECK: bufferization.materialize_in_destination
