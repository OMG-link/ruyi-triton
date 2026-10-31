// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental  %s | FileCheck %s
module {
  tt.func public @minmax_olt(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf olt, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}


// -----

module {
  tt.func public @minmax_ole(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf ole, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}


// -----

module {
  tt.func public @minmax_ogt(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf ogt, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}


// -----

module {
  tt.func public @minmax_oge(%arg0: !tt.ptr<f32>, %arg1: f32, %arg2: f32) {
    %0 = arith.cmpf oge, %arg1, %arg2 : f32
    %1 = arith.select %0, %arg1, %arg2 : f32
    tt.store %arg0, %1 : !tt.ptr<f32>
    tt.return
  }
}

// CHECK lines should be minimized and named to reflect the test’s intent.
// CHECK-LABEL:   func.func @minmax_olt(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {llvm.bareptr} {
// CHECK:           %[[MINIMUMF_0:.*]] = arith.minimumf %[[ARG1]], %[[ARG2]] : f32
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1]>>
// CHECK:           affine.store %[[MINIMUMF_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1]>>
// CHECK:           return
// CHECK:         }
// CHECK-LABEL:   func.func @minmax_ole(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {llvm.bareptr} {
// CHECK:           %[[MINIMUMF_0:.*]] = arith.minimumf %[[ARG1]], %[[ARG2]] : f32
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1]>>
// CHECK:           affine.store %[[MINIMUMF_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1]>>
// CHECK:           return
// CHECK:         }
// CHECK-LABEL:   func.func @minmax_ogt(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {llvm.bareptr} {
// CHECK:           %[[MAXIMUMF_0:.*]] = arith.maximumf %[[ARG1]], %[[ARG2]] : f32
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1]>>
// CHECK:           affine.store %[[MAXIMUMF_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1]>>
// CHECK:           return
// CHECK:         }
// CHECK-LABEL:   func.func @minmax_oge(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: f32,
// CHECK-SAME:      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG7:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG8:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {llvm.bareptr} {
// CHECK:           %[[MAXIMUMF_0:.*]] = arith.maximumf %[[ARG1]], %[[ARG2]] : f32
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<f32> to memref<1xf32, strided<[1]>>
// CHECK:           affine.store %[[MAXIMUMF_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xf32, strided<[1]>>
// CHECK:           return
// CHECK:         }
