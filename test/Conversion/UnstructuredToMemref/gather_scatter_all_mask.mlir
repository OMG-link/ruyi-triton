// RUN: triton-shared-opt --triton-to-unstructured --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @masked_gather_scatter(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %cst = arith.constant dense<9.900000e+01> : tensor<4xf32>
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst_0 = arith.constant dense<4> : tensor<4xi32>
    %cst_1 = arith.constant dense<64> : tensor<4xi32>
    %cst_2 = arith.constant dense<3> : tensor<4xi32>
    %0 = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<4x!tt.ptr<f32>>
    %3:2 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %0, %arg4 = %0) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
      %4 = arith.divsi %arg3, %cst_2 : tensor<4xi32>
      %5 = tt.splat %arg2 : i32 -> tensor<4xi32>
      %6 = arith.addi %4, %5 : tensor<4xi32>
      %7 = arith.cmpi slt, %6, %cst_1 : tensor<4xi32>
      %8 = tt.addptr %1, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      %9 = tt.load %8, %7, %cst : tensor<4x!tt.ptr<f32>>
      %10 = tt.addptr %2, %6 : tensor<4x!tt.ptr<f32>>, tensor<4xi32>
      tt.store %10, %9, %7 : tensor<4x!tt.ptr<f32>>
      %11 = arith.addi %6, %cst_0 : tensor<4xi32>
      %12 = arith.addi %arg4, %cst_0 : tensor<4xi32>
      scf.yield %11, %12 : tensor<4xi32>, tensor<4xi32>
    }
    tt.return
  }
}

// CHECK lines should be minimized and named to reflect the test’s intent.
// CHECK-LABEL:   tt.func public @masked_gather_scatter(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 4 : index
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant dense<3> : tensor<4xi32>
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant dense<64> : tensor<4xi32>
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant dense<4> : tensor<4xi32>
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_8:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_9:.*]] = arith.constant 9.900000e+01 : f32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 4 : i32, start = 0 : i32} : tensor<4xi32>
// CHECK:           %[[FOR_0:.*]]:2 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_6]] to %[[CONSTANT_7]] step %[[CONSTANT_8]] iter_args(%[[VAL_1:.*]] = %[[MAKE_RANGE_0]], %[[VAL_2:.*]] = %[[MAKE_RANGE_0]]) -> (tensor<4xi32>, tensor<4xi32>)  : i32 {
// CHECK:             %[[DIVSI_0:.*]] = arith.divsi %[[VAL_1]], %[[CONSTANT_3]] : tensor<4xi32>
// CHECK:             %[[SPLAT_0:.*]] = tt.splat %[[VAL_0]] : i32 -> tensor<4xi32>
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[DIVSI_0]], %[[SPLAT_0]] : tensor<4xi32>
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[ADDI_0]], %[[CONSTANT_4]] : tensor<4xi32>
// CHECK:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<4xf32>
// CHECK:             %[[FOR_1:.*]] = scf.for %[[VAL_3:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] iter_args(%[[VAL_4:.*]] = %[[EMPTY_0]]) -> (tensor<4xf32>) {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[ADDI_0]]{{\[}}%[[VAL_3]]] : tensor<4xi32>
// CHECK:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[CMPI_0]]{{\[}}%[[VAL_3]]] : tensor<4xi1>
// CHECK:               %[[IF_0:.*]] = scf.if %[[EXTRACT_1]] -> (f32) {
// CHECK:                 %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [], strides: [] : memref<*xf32> to memref<f32, strided<[], offset: ?>>
// CHECK:                 %[[LOAD_0:.*]] = memref.load %[[REINTERPRET_CAST_0]][] : memref<f32, strided<[], offset: ?>>
// CHECK:                 scf.yield %[[LOAD_0]] : f32
// CHECK:               } else {
// CHECK:                 scf.yield %[[CONSTANT_9]] : f32
// CHECK:               }
// CHECK:               %[[INSERT_0:.*]] = tensor.insert %[[IF_0]] into %[[VAL_4]]{{\[}}%[[VAL_3]]] : tensor<4xf32>
// CHECK:               scf.yield %[[INSERT_0]] : tensor<4xf32>
// CHECK:             }
// CHECK:             scf.for %[[VAL_5:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_2]] step %[[CONSTANT_0]] {
// CHECK:               %[[EXTRACT_2:.*]] = tensor.extract %[[ADDI_0]]{{\[}}%[[VAL_5]]] : tensor<4xi32>
// CHECK:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[EXTRACT_2]] : i32 to index
// CHECK:               %[[EXTRACT_3:.*]] = tensor.extract %[[FOR_1]]{{\[}}%[[VAL_5]]] : tensor<4xf32>
// CHECK:               %[[EXTRACT_4:.*]] = tensor.extract %[[CMPI_0]]{{\[}}%[[VAL_5]]] : tensor<4xi1>
// CHECK:               scf.if %[[EXTRACT_4]] {
// CHECK:                 %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[INDEX_CAST_1]]], sizes: [], strides: [] : memref<*xf32> to memref<f32, strided<[], offset: ?>>
// CHECK:                 memref.store %[[EXTRACT_3]], %[[REINTERPRET_CAST_1]][] : memref<f32, strided<[], offset: ?>>
// CHECK:               }
// CHECK:             }
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[ADDI_0]], %[[CONSTANT_5]] : tensor<4xi32>
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_5]] : tensor<4xi32>
// CHECK:             scf.yield %[[ADDI_1]], %[[ADDI_2]] : tensor<4xi32>, tensor<4xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
