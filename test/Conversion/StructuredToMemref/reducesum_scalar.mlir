// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
module {
  tt.func @kernel(%afloat : !tt.ptr<bf16>, %res : !tt.ptr<bf16>)
  {
    %0 = tt.make_range {end = 128 : i32, start = 0 : i32} : tensor<128xi32>
    %1 = tt.splat %afloat : !tt.ptr<bf16> -> tensor<128x!tt.ptr<bf16>>
    %2 = tt.addptr %1, %0 : tensor<128x!tt.ptr<bf16>>, tensor<128xi32>
    %afm = tt.load %2 : tensor<128x!tt.ptr<bf16>>
    %3 = "tt.reduce"(%afm) ({
    ^bb0(%arg5: bf16, %arg6: bf16):
      %21 = arith.addf %arg5, %arg6 : bf16
      tt.reduce.return %21 : bf16
    }) {axis = 0 : i32} : (tensor<128xbf16>) -> bf16
    tt.store %res, %3 : !tt.ptr<bf16>
    tt.return
  }
}

// CHECK lines should be minimized and named to reflect the test’s intent.
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<bf16>,
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<bf16>,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {llvm.bareptr} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [128], strides: [1] : memref<bf16> to memref<128xbf16, strided<[1]>>
// CHECK:           %[[ALLOC_0:.*]] = memref.alloc() : memref<128xbf16>
// CHECK:           memref.copy %[[REINTERPRET_CAST_0]], %[[ALLOC_0]] : memref<128xbf16, strided<[1]>> to memref<128xbf16>
// CHECK:           %[[TO_TENSOR_0:.*]] = bufferization.to_tensor %[[ALLOC_0]] restrict writable : memref<128xbf16> to tensor<128xbf16>
// CHECK:           %[[EMPTY_0:.*]] = tensor.empty() : tensor<f32>
// CHECK:           %[[FILL_0:.*]] = linalg.fill ins(%[[CONSTANT_0]] : f32) outs(%[[EMPTY_0]] : tensor<f32>) -> tensor<f32>
// CHECK:           %[[REDUCE_0:.*]] = linalg.reduce ins(%[[TO_TENSOR_0]] : tensor<128xbf16>) outs(%[[FILL_0]] : tensor<f32>) dimensions = [0]
// CHECK:             (%[[VAL_0:.*]]: bf16, %[[VAL_1:.*]]: f32) {
// CHECK:               %[[EXTF_0:.*]] = arith.extf %[[VAL_0]] : bf16 to f32
// CHECK:               %[[ADDF_0:.*]] = arith.addf %[[EXTF_0]], %[[VAL_1]] : f32
// CHECK:               linalg.yield %[[ADDF_0]] : f32
// CHECK:             }
// CHECK:           %[[EXTRACT_0:.*]] = tensor.extract %[[REDUCE_0]][] : tensor<f32>
// CHECK:           %[[TRUNCF_0:.*]] = arith.truncf %[[EXTRACT_0]] : f32 to bf16
// CHECK:           %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[ARG1]] to offset: [0], sizes: [1], strides: [1] : memref<bf16> to memref<1xbf16, strided<[1]>>
// CHECK:           affine.store %[[TRUNCF_0]], %[[REINTERPRET_CAST_1]][0] : memref<1xbf16, strided<[1]>>
// CHECK:           return
// CHECK:         }
