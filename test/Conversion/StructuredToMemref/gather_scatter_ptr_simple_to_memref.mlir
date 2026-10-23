// RUN: triton-shared-opt  --structured-to-memref --canonicalize --cse %s | FileCheck %s

// CHECK: #[[$GATHER_MAP:.+]] = affine_map<(d0, d1) -> (d0)>
// CHECK: #[[$OUTPUT_MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>

// CHECK-LABEL:   tt.func public @row_gather(
// CHECK-SAME:                               %[[VAL_0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:                               %[[VAL_1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<i32>,
// CHECK-SAME:                               %[[VAL_2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK-DAG:       %[[BASE:.*]] = builtin.unrealized_conversion_cast %[[VAL_0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK-DAG:       %[[INDEX_BASE:.*]] = builtin.unrealized_conversion_cast %[[VAL_1]] : !tt.ptr<i32> to memref<*xi32>
// CHECK:           %[[SOURCE_LENGTH:.*]] = memref.dim %[[BASE]], %{{.*}} : memref<*xf32>
// CHECK:           %[[SOURCE_MEMREF:.*]] = memref.reinterpret_cast %[[BASE]] to offset: [0], sizes: [%[[SOURCE_LENGTH]]], strides: [1] : memref<*xf32> to memref<?xf32>
// CHECK:           %[[SOURCE_TENSOR:.*]] = bufferization.to_tensor %[[SOURCE_MEMREF]] restrict : memref<?xf32> to tensor<?xf32>
// CHECK:           %[[EMPTY:.*]] = tensor.empty() : tensor<8x8xf32>
// CHECK:           %[[RESULT:.*]] = linalg.generic {indexing_maps = [#[[$GATHER_MAP]], #[[$OUTPUT_MAP]]], iterator_types = ["parallel", "parallel"]} ins(%{{.*}} : tensor<8xi32>) outs(%[[EMPTY]] : tensor<8x8xf32>) {
// CHECK:             ^bb0(%[[GATHER_INDEX:.*]]: i32, %{{.*}}: f32):
// CHECK:               %[[GATHER_INDEX_INDEX:.*]] = arith.index_cast %[[GATHER_INDEX]] : i32 to index
// CHECK:               %{{.*}} = linalg.index 1 : index
// CHECK:               %{{.*}} = arith.addi %[[GATHER_INDEX_INDEX]], %{{.*}} : index
// CHECK:               %{{.*}} = tensor.extract %[[SOURCE_TENSOR]][%{{.*}}] : tensor<?xf32>
// CHECK:               linalg.yield %{{.*}} : f32
// CHECK:           } -> tensor<8x8xf32>
// CHECK:           %[[DEST:.*]] = builtin.unrealized_conversion_cast %[[VAL_2]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[DEST_MEMREF:.*]] = memref.reinterpret_cast %[[DEST]] to offset: [0], sizes: [8, 8], strides: [8, 1] : memref<*xf32> to memref<8x8xf32, strided<[8, 1]>>
// CHECK:           bufferization.materialize_in_destination %[[RESULT]] in writable %[[DEST_MEMREF]] : (tensor<8x8xf32>, memref<8x8xf32, strided<[8, 1]>>) -> ()


module {
  tt.func public @row_gather(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<8> : tensor<8xi32>
    %c8 = arith.constant 8 : index
    %cst_0 = arith.constant dense<3> : tensor<8xi32>
    %cst_1 = arith.constant dense<5> : tensor<8xi32>
    %0 = tts.make_tptr %arg1 to sizes: [8], strides: [1], offsets: [0], shape: [0], order: [] : <i32> to tensor<8x!tt.ptr<i32>>
    %1 = "tts.load"(%0) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<8x!tt.ptr<i32>>) -> tensor<8xi32>
    %2 = arith.muli %1, %cst_0 : tensor<8xi32>
    %3 = arith.remsi %2, %cst_1 : tensor<8xi32>
    %4 = arith.muli %3, %cst : tensor<8xi32>
    %5 = tts.make_gather_scatter_tptr %arg0 to sizes: [8, 8] gather_scatter_dim: 0 gather_scatter_offset: %4, strides: [1, 1], offsets: [0, 0] : tensor<8xi32> <f32> to tensor<8x8x!tt.ptr<f32>>
    %6 = "tts.load"(%5) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<8x8x!tt.ptr<f32>>) -> tensor<8x8xf32>
    %7 = tts.make_tptr %arg2 to sizes: [8, 8], strides: [%c8, 1], offsets: [0, 0], shape: [0, 0], order: [] : <f32> to tensor<8x8x!tt.ptr<f32>>
    "tts.store"(%7, %6) <{static_mask_dims = array<i64>}> : (tensor<8x8x!tt.ptr<f32>>, tensor<8x8xf32>) -> ()
    tt.return
  }
}
