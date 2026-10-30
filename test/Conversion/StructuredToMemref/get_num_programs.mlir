// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s

// A zero-offset singleton store must work even when pointer arithmetic folds
// away. The source address carries no allocation shape metadata.
module {
  tt.func public @num_programs(%arg0: !tt.ptr<i32>) {
    %0 = tt.get_num_programs x : i32
    %1 = tt.get_num_programs y : i32
    %2 = tt.get_num_programs z : i32
    %3 = tt.make_range {end = 1 : i32, start = 0 : i32} : tensor<1xi32>
    %4 = tt.make_range {end = 2 : i32, start = 1 : i32} : tensor<1xi32>
    %5 = tt.make_range {end = 3 : i32, start = 2 : i32} : tensor<1xi32>
    %6 = tt.splat %arg0 : !tt.ptr<i32> -> tensor<1x!tt.ptr<i32>>
    %7 = tt.addptr %6, %3 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %8 = tt.splat %0 : i32 -> tensor<1xi32>
    tt.store %7, %8 : tensor<1x!tt.ptr<i32>>
    %9 = tt.addptr %6, %4 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %10 = tt.splat %1 : i32 -> tensor<1xi32>
    tt.store %9, %10 : tensor<1x!tt.ptr<i32>>
    %11 = tt.addptr %6, %5 : tensor<1x!tt.ptr<i32>>, tensor<1xi32>
    %12 = tt.splat %2 : i32 -> tensor<1xi32>
    tt.store %11, %12 : tensor<1x!tt.ptr<i32>>
    tt.return
  }
}

// CHECK-LABEL: func.func @num_programs(
// CHECK-SAME: [[BASE:%[a-zA-Z0-9_.$-]+]]: memref<i32>, [[NX:%[a-zA-Z0-9_.$-]+]]: i32, [[NY:%[a-zA-Z0-9_.$-]+]]: i32, [[NZ:%[a-zA-Z0-9_.$-]+]]: i32, {{.*}}) attributes {llvm.bareptr}
// CHECK: [[FIRST:%[a-zA-Z0-9_.$-]+]] = memref.reinterpret_cast [[BASE]] to offset: [0], sizes: [], strides: [] : memref<i32> to memref<i32, strided<[]>>
// CHECK: memref.store [[NX]], [[FIRST]][] : memref<i32, strided<[]>>
// CHECK: [[TENSOR:%[a-zA-Z0-9_.$-]+]] = tensor.empty() : tensor<1xi32>
// CHECK: [[Y:%[a-zA-Z0-9_.$-]+]] = linalg.fill ins([[NY]] : i32) outs([[TENSOR]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK: [[SECOND:%[a-zA-Z0-9_.$-]+]] = memref.reinterpret_cast [[BASE]] to offset: [1], sizes: [1], strides: [1] : memref<i32> to memref<1xi32, strided<[1], offset: 1>>
// CHECK: [[SECOND_VIEW:%[a-zA-Z0-9_.$-]+]] = memref.cast [[SECOND]] : memref<1xi32, strided<[1], offset: 1>> to memref<1xi32, strided<[1], offset: ?>>
// CHECK: bufferization.materialize_in_destination [[Y]] in writable [[SECOND_VIEW]]
// CHECK: [[Z:%[a-zA-Z0-9_.$-]+]] = linalg.fill ins([[NZ]] : i32) outs([[TENSOR]] : tensor<1xi32>) -> tensor<1xi32>
// CHECK: [[THIRD:%[a-zA-Z0-9_.$-]+]] = memref.reinterpret_cast [[BASE]] to offset: [2], sizes: [1], strides: [1] : memref<i32> to memref<1xi32, strided<[1], offset: 2>>
// CHECK: [[THIRD_VIEW:%[a-zA-Z0-9_.$-]+]] = memref.cast [[THIRD]] : memref<1xi32, strided<[1], offset: 2>> to memref<1xi32, strided<[1], offset: ?>>
// CHECK: bufferization.materialize_in_destination [[Z]] in writable [[THIRD_VIEW]]
// CHECK: return
