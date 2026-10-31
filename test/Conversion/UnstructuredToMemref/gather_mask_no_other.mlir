// RUN: triton-shared-opt --triton-to-unstructured --canonicalize --unstructured-to-memref --canonicalize %s | FileCheck %s

module {
  tt.func public @gather_simple_mask_no_other(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) attributes {noinline = false} {
    %c1_i32 = arith.constant 1 : i32
    %c2_i32 = arith.constant 2 : i32
    %c0_i32 = arith.constant 0 : i32
    %cst = arith.constant dense<64> : tensor<64xi32>
    %c16_i32 = arith.constant 16 : i32
    %cst_0 = arith.constant dense<4> : tensor<64xi32>
    %c8_i32 = arith.constant 8 : i32
    %0 = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %1 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %2 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3:3 = scf.for %arg2 = %c0_i32 to %c2_i32 step %c1_i32 iter_args(%arg3 = %c8_i32, %arg4 = %0, %arg5 = %0) -> (i32, tensor<64xi32>, tensor<64xi32>)  : i32 {
      %4 = arith.divsi %arg4, %cst_0 : tensor<64xi32>
      %5 = tt.splat %arg3 : i32 -> tensor<64xi32>
      %6 = arith.cmpi slt, %4, %5 : tensor<64xi32>
      %7 = tt.addptr %1, %4 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %8 = tt.load %7, %6 : tensor<64x!tt.ptr<f32>>
      %9 = tt.addptr %2, %arg5 : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %9, %8 : tensor<64x!tt.ptr<f32>>
      %10 = arith.addi %arg3, %c16_i32 : i32
      %11 = arith.addi %arg4, %cst : tensor<64xi32>
      %12 = arith.addi %arg5, %cst : tensor<64xi32>
      scf.yield %10, %11, %12 : i32, tensor<64xi32>, tensor<64xi32>
    }
    tt.return
  }
}

// CHECK lines should be minimized and named to reflect the test’s intent.
// CHECK-LABEL:   tt.func public @gather_simple_mask_no_other(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: !tt.ptr<f32>) attributes {noinline = false} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 1 : index
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 0 : index
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0.000000e+00 : f32
// CHECK:           %[[CONSTANT_3:.*]] = arith.constant 64 : index
// CHECK:           %[[CONSTANT_4:.*]] = arith.constant 8 : i32
// CHECK:           %[[CONSTANT_5:.*]] = arith.constant dense<4> : tensor<64xi32>
// CHECK:           %[[CONSTANT_6:.*]] = arith.constant 16 : i32
// CHECK:           %[[CONSTANT_7:.*]] = arith.constant dense<64> : tensor<64xi32>
// CHECK:           %[[CONSTANT_8:.*]] = arith.constant 0 : i32
// CHECK:           %[[CONSTANT_9:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_10:.*]] = arith.constant 1 : i32
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_0:.*]] = builtin.unrealized_conversion_cast %[[ARG1]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[UNREALIZED_CONVERSION_CAST_1:.*]] = builtin.unrealized_conversion_cast %[[ARG0]] : !tt.ptr<f32> to memref<*xf32>
// CHECK:           %[[MAKE_RANGE_0:.*]] = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
// CHECK:           %[[FOR_0:.*]]:3 = scf.for %[[VAL_0:.*]] = %[[CONSTANT_8]] to %[[CONSTANT_9]] step %[[CONSTANT_10]] iter_args(%[[VAL_1:.*]] = %[[CONSTANT_4]], %[[VAL_2:.*]] = %[[MAKE_RANGE_0]], %[[VAL_3:.*]] = %[[MAKE_RANGE_0]]) -> (i32, tensor<64xi32>, tensor<64xi32>)  : i32 {
// CHECK:             %[[DIVSI_0:.*]] = arith.divsi %[[VAL_2]], %[[CONSTANT_5]] : tensor<64xi32>
// CHECK:             %[[SPLAT_0:.*]] = tt.splat %[[VAL_1]] : i32 -> tensor<64xi32>
// CHECK:             %[[CMPI_0:.*]] = arith.cmpi slt, %[[DIVSI_0]], %[[SPLAT_0]] : tensor<64xi32>
// CHECK:             %[[EMPTY_0:.*]] = tensor.empty() : tensor<64xf32>
// CHECK:             %[[FOR_1:.*]] = scf.for %[[VAL_4:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_3]] step %[[CONSTANT_0]] iter_args(%[[VAL_5:.*]] = %[[EMPTY_0]]) -> (tensor<64xf32>) {
// CHECK:               %[[EXTRACT_0:.*]] = tensor.extract %[[DIVSI_0]]{{\[}}%[[VAL_4]]] : tensor<64xi32>
// CHECK:               %[[INDEX_CAST_0:.*]] = arith.index_cast %[[EXTRACT_0]] : i32 to index
// CHECK:               %[[EXTRACT_1:.*]] = tensor.extract %[[CMPI_0]]{{\[}}%[[VAL_4]]] : tensor<64xi1>
// CHECK:               %[[IF_0:.*]] = scf.if %[[EXTRACT_1]] -> (f32) {
// CHECK:                 %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_1]] to offset: {{\[}}%[[INDEX_CAST_0]]], sizes: [], strides: [] : memref<*xf32> to memref<f32, strided<[], offset: ?>>
// CHECK:                 %[[LOAD_0:.*]] = memref.load %[[REINTERPRET_CAST_0]][] : memref<f32, strided<[], offset: ?>>
// CHECK:                 scf.yield %[[LOAD_0]] : f32
// CHECK:               } else {
// CHECK:                 scf.yield %[[CONSTANT_2]] : f32
// CHECK:               }
// CHECK:               %[[INSERT_0:.*]] = tensor.insert %[[IF_0]] into %[[VAL_5]]{{\[}}%[[VAL_4]]] : tensor<64xf32>
// CHECK:               scf.yield %[[INSERT_0]] : tensor<64xf32>
// CHECK:             }
// CHECK:             scf.for %[[VAL_6:.*]] = %[[CONSTANT_1]] to %[[CONSTANT_3]] step %[[CONSTANT_0]] {
// CHECK:               %[[EXTRACT_2:.*]] = tensor.extract %[[VAL_3]]{{\[}}%[[VAL_6]]] : tensor<64xi32>
// CHECK:               %[[INDEX_CAST_1:.*]] = arith.index_cast %[[EXTRACT_2]] : i32 to index
// CHECK:               %[[EXTRACT_3:.*]] = tensor.extract %[[FOR_1]]{{\[}}%[[VAL_6]]] : tensor<64xf32>
// CHECK:               %[[REINTERPRET_CAST_1:.*]] = memref.reinterpret_cast %[[UNREALIZED_CONVERSION_CAST_0]] to offset: {{\[}}%[[INDEX_CAST_1]]], sizes: [], strides: [] : memref<*xf32> to memref<f32, strided<[], offset: ?>>
// CHECK:               memref.store %[[EXTRACT_3]], %[[REINTERPRET_CAST_1]][] : memref<f32, strided<[], offset: ?>>
// CHECK:             }
// CHECK:             %[[ADDI_0:.*]] = arith.addi %[[VAL_1]], %[[CONSTANT_6]] : i32
// CHECK:             %[[ADDI_1:.*]] = arith.addi %[[VAL_2]], %[[CONSTANT_7]] : tensor<64xi32>
// CHECK:             %[[ADDI_2:.*]] = arith.addi %[[VAL_3]], %[[CONSTANT_7]] : tensor<64xi32>
// CHECK:             scf.yield %[[ADDI_0]], %[[ADDI_1]], %[[ADDI_2]] : i32, tensor<64xi32>, tensor<64xi32>
// CHECK:           }
// CHECK:           tt.return
// CHECK:         }
