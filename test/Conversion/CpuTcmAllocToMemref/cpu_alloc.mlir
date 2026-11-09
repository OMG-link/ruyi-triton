// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s --check-prefix=NONE

// triton.language.extra.cpu.alloc: the pointer returned by the runtime symbol
// __triton_cpu_tcm_alloc becomes a memref.alloc marked cpu.tcm, accessed
// through memref.view; no tptr op and no call remain.

module {
  // p = cpu.alloc((64,), tl.float32); tl.store(p + offs, x * 2);
  // tl.store(out + offs, tl.load(p + offs))
  tt.func public @block(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<f32>) {
    %cst = arith.constant dense<2.000000e+00> : tensor<64xf32>
    %bytes = arith.constant 256 : i64
    %align = arith.constant 128 : i64
    %a = tt.extern_elementwise %bytes, %align {libname = "", libpath = "", pure = false, symbol = "__triton_cpu_tcm_alloc"} : (i64, i64) -> i64
    %p = tt.int_to_ptr %a : i64 -> !tt.ptr<f32>
    %offs = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %0 = tt.splat %p : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %1 = tt.addptr %0, %offs : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %2 = tt.splat %x_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %3 = tt.addptr %2, %offs : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %4 = tt.load %3 : tensor<64x!tt.ptr<f32>>
    %5 = arith.mulf %4, %cst : tensor<64xf32>
    tt.store %1, %5 : tensor<64x!tt.ptr<f32>>
    %6 = tt.splat %out_ptr : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %7 = tt.addptr %6, %offs : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %8 = tt.load %1 : tensor<64x!tt.ptr<f32>>
    tt.store %7, %8 : tensor<64x!tt.ptr<f32>>
    tt.return
  }

  // Two buffers and a scalar access at a constant offset.
  tt.func public @scalar(%x_ptr: !tt.ptr<f32>, %out_ptr: !tt.ptr<i32>) {
    %c3_i32 = arith.constant 3 : i32
    %c7_i32 = arith.constant 7 : i32
    %bytes0 = arith.constant 16 : i64
    %bytes1 = arith.constant 4 : i64
    %align = arith.constant 128 : i64
    %a0 = tt.extern_elementwise %bytes0, %align {libname = "", libpath = "", pure = false, symbol = "__triton_cpu_tcm_alloc"} : (i64, i64) -> i64
    %a1 = tt.extern_elementwise %bytes1, %align {libname = "", libpath = "", pure = false, symbol = "__triton_cpu_tcm_alloc"} : (i64, i64) -> i64
    %p0 = tt.int_to_ptr %a0 : i64 -> !tt.ptr<f32>
    %p1 = tt.int_to_ptr %a1 : i64 -> !tt.ptr<i32>
    %q0 = tt.addptr %p0, %c3_i32 : !tt.ptr<f32>, i32
    %x = tt.load %x_ptr : !tt.ptr<f32>
    tt.store %q0, %x : !tt.ptr<f32>
    tt.store %p1, %c7_i32 : !tt.ptr<i32>
    %y = tt.load %p1 : !tt.ptr<i32>
    tt.store %out_ptr, %y : !tt.ptr<i32>
    tt.return
  }
}

// NONE-NOT:    tptr.
// NONE-NOT:    __triton_cpu_tcm_alloc

// CHECK-LABEL: func.func @block
// CHECK:         [[BUF:%.+]] = memref.alloc() {alignment = 128 : i64, cpu.tcm} : memref<256xi8>
// CHECK:         [[VIEW:%.+]] = memref.view [[BUF]][%{{.+}}][] : memref<256xi8> to memref<1xf32>
// CHECK:         memref.reinterpret_cast [[VIEW]] to offset: [0], sizes: [64], strides: [1]

// CHECK-LABEL: func.func @scalar
// CHECK-DAG:     [[BUF0:%.+]] = memref.alloc() {alignment = 128 : i64, cpu.tcm} : memref<16xi8>
// CHECK-DAG:     [[BUF1:%.+]] = memref.alloc() {alignment = 128 : i64, cpu.tcm} : memref<4xi8>
// CHECK-DAG:     memref.view [[BUF0]][%{{.+}}][] : memref<16xi8> to memref<1xf32>
// CHECK-DAG:     memref.view [[BUF1]][%{{.+}}][] : memref<4xi8> to memref<1xi32>

