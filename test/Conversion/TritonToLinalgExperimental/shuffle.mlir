// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
//
// tt.dev.shuffle becomes cpu.shuffle on the same tensors with the same mask,
// which buddy-mlir's tile tensor-to-vector turns into vector.shuffle.

module {
  // CHECK-LABEL: func.func @slide
  // CHECK: cpu.shuffle %{{[a-z0-9_]+}}, %{{[a-z0-9_]+}} [1, 2, 3, {{.*}}, 63, 64] : tensor<64xf16>, tensor<64xf16> -> tensor<64xf16>
  // CHECK-NOT: tt.dev.shuffle
  tt.func public @slide(%in: !tt.ptr<f16>, %out: !tt.ptr<f16>) {
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %in_s = tt.splat %in : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
    %in_p = tt.addptr %in_s, %range : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
    %x = tt.load %in_p : tensor<64x!tt.ptr<f16>>
    %zero = arith.constant dense<0.0> : tensor<64xf16>
    %y = tt.dev.shuffle %x, %zero [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64] : tensor<64xf16>, tensor<64xf16> -> tensor<64xf16>
    %out_s = tt.splat %out : !tt.ptr<f16> -> tensor<64x!tt.ptr<f16>>
    %out_p = tt.addptr %out_s, %range : tensor<64x!tt.ptr<f16>>, tensor<64xi32>
    tt.store %out_p, %y : tensor<64x!tt.ptr<f16>>
    tt.return
  }
}
