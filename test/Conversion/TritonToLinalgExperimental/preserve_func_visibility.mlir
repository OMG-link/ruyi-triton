// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  tt.func public @kernel(%arg0: !tt.ptr<i32> {tt.divisibility = 16 : i32}) attributes {noinline = false} {
    %c0_i32 = arith.constant 0 : i32
    %0 = tt.get_program_id x : i32
    %1 = tt.load %arg0 : !tt.ptr<i32>
    %2 = arith.cmpi eq, %0, %c0_i32 : i32
    %3 = scf.if %2 -> (i32) {
      %4 = tt.call @test_core.add_fn_return__i32_i32__(%1, %0) : (i32, i32) -> i32
      scf.yield %4 : i32
    } else {
      scf.yield %1 : i32
    }
    tt.store %arg0, %3 : !tt.ptr<i32>
    tt.return
  }
  tt.func private @test_core.add_fn_return__i32_i32__(%arg0: i32, %arg1: i32) -> i32 attributes {noinline = false} {
    %c2_i32 = arith.constant 2 : i32
    %c1_i32 = arith.constant 1 : i32
    %c0_i32 = arith.constant 0 : i32
    %0 = arith.cmpi eq, %arg1, %c0_i32 : i32
    cf.cond_br %0, ^bb1(%c1_i32 : i32), ^bb1(%c2_i32 : i32)
  ^bb1(%1: i32):  // 2 preds: ^bb0, ^bb0
    %2 = arith.addi %arg0, %1 : i32
    tt.return %2 : i32
  }
}

// Public is implicit

// CHECK lines should be minimized and named to reflect the test’s intent.
// CHECK-LABEL:   func.func @kernel(
// CHECK-SAME:                      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: memref<i32> {tt.divisibility = 16 : i32},
// CHECK-SAME:                      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG2:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG3:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG4:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG5:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:                      %[[ARG6:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) attributes {llvm.bareptr} {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 0 : i32
// CHECK:           %[[REINTERPRET_CAST_0:.*]] = memref.reinterpret_cast %[[ARG0]] to offset: [0], sizes: [1], strides: [1] : memref<i32> to memref<1xi32, strided<[1]>>
// CHECK:           %[[LOAD_0:.*]] = affine.load %[[REINTERPRET_CAST_0]][0] : memref<1xi32, strided<[1]>>
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[ARG4]], %[[CONSTANT_0]] : i32
// CHECK:           %[[IF_0:.*]] = scf.if %[[CMPI_0]] -> (i32) {
// CHECK:             %[[CALL_0:.*]] = func.call @test_core.add_fn_return__i32_i32__(%[[LOAD_0]], %[[ARG4]]) : (i32, i32) -> i32
// CHECK:             scf.yield %[[CALL_0]] : i32
// CHECK:           } else {
// CHECK:             scf.yield %[[LOAD_0]] : i32
// CHECK:           }
// CHECK:           affine.store %[[IF_0]], %[[REINTERPRET_CAST_0]][0] : memref<1xi32, strided<[1]>>
// CHECK:           return
// CHECK:         }
// CHECK-LABEL:   func.func private @test_core.add_fn_return__i32_i32__(
// CHECK-SAME:      %[[ARG0:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32,
// CHECK-SAME:      %[[ARG1:[0-9]+|[a-zA-Z$._-][a-zA-Z0-9$._-]*]]: i32) -> i32 {
// CHECK:           %[[CONSTANT_0:.*]] = arith.constant 2 : i32
// CHECK:           %[[CONSTANT_1:.*]] = arith.constant 1 : i32
// CHECK:           %[[CONSTANT_2:.*]] = arith.constant 0 : i32
// CHECK:           %[[CMPI_0:.*]] = arith.cmpi eq, %[[ARG1]], %[[CONSTANT_2]] : i32
// CHECK:           %[[SELECT_0:.*]] = arith.select %[[CMPI_0]], %[[CONSTANT_1]], %[[CONSTANT_0]] : i32
// CHECK:           %[[ADDI_0:.*]] = arith.addi %[[ARG0]], %[[SELECT_0]] : i32
// CHECK:           return %[[ADDI_0]] : i32
// CHECK:         }
