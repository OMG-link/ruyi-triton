// RUN: triton-shared-opt --split-input-file --verify-diagnostics --cpu-tcm-alloc-to-memref %s | FileCheck %s

// --cpu-tcm-alloc-to-memref on the tptr ops left by the experimental pipeline:
// pointer arithmetic becomes a byte offset of memref.view, ptrtoint the
// address of the buffer plus that offset.

module {
  func.func private @__triton_cpu_tcm_alloc(i64, i64) -> i64
  func.func @chain(%k: i32) -> (f32, i64) {
    %c0 = arith.constant 0 : index
    %bytes = arith.constant 1024 : i64
    %align = arith.constant 128 : i64
    %a = func.call @__triton_cpu_tcm_alloc(%bytes, %align) : (i64, i64) -> i64
    %p = tptr.inttoptr %a : i64 to !ptr.ptr<#tptr.default_memory_space>
    %size = tptr.type_offset f32 : i32
    %off = arith.muli %k, %size : i32
    %q = tptr.ptradd %p %off : !ptr.ptr<#tptr.default_memory_space>, i32 to !ptr.ptr<#tptr.default_memory_space>
    %m = tptr.to_memref %q : !ptr.ptr<#tptr.default_memory_space> to memref<1xf32>
    %v = memref.load %m[%c0] : memref<1xf32>
    %i = tptr.ptrtoint %q : !ptr.ptr<#tptr.default_memory_space> to i64
    return %v, %i : f32, i64
  }
}

// CHECK-NOT:     func.func private @__triton_cpu_tcm_alloc
// CHECK-LABEL:   func.func @chain
// CHECK-SAME:      ([[K:%.+]]: i32)
// CHECK:           [[BUF:%.+]] = memref.alloc() {alignment = 128 : i64, cpu.tcm} : memref<1024xi8>
// CHECK-DAG:       [[SIZE:%.+]] = arith.constant 4 : index
// CHECK-DAG:       [[COUNT:%.+]] = arith.index_cast [[K]] : i32 to index
// CHECK:           [[SHIFT:%.+]] = arith.muli [[COUNT]], [[SIZE]] : index
// CHECK:           [[VIEW:%.+]] = memref.view [[BUF]]{{\[}}[[SHIFT]]][] : memref<1024xi8> to memref<1xf32>
// CHECK:           memref.load [[VIEW]]
// CHECK:           [[BASE:%.+]] = memref.extract_aligned_pointer_as_index [[BUF]]
// CHECK:           [[ADDR:%.+]] = arith.addi [[BASE]], [[SHIFT]] : index
// CHECK:           arith.index_cast [[ADDR]] : index to i64
// CHECK-NOT:       tptr.inttoptr
// CHECK-NOT:       tptr.ptradd
// CHECK-NOT:       tptr.to_memref
// CHECK-NOT:       tptr.ptrtoint
// CHECK-NOT:       call

// -----

module {
  func.func private @__triton_cpu_tcm_alloc(i64, i64) -> i64
  func.func @dynamic_size(%n: i64) -> i64 {
    %align = arith.constant 128 : i64
    // expected-error @+1 {{cpu.alloc: size and alignment must be compile-time constants}}
    %a = func.call @__triton_cpu_tcm_alloc(%n, %align) : (i64, i64) -> i64
    return %a : i64
  }
}

// -----

module {
  func.func private @__triton_cpu_tcm_alloc(i64, i64) -> i64
  func.func private @use(!ptr.ptr<#tptr.default_memory_space>)
  func.func @escaping() {
    %bytes = arith.constant 64 : i64
    %align = arith.constant 128 : i64
    // expected-note @+1 {{buffer allocated here}}
    %a = func.call @__triton_cpu_tcm_alloc(%bytes, %align) : (i64, i64) -> i64
    %p = tptr.inttoptr %a : i64 to !ptr.ptr<#tptr.default_memory_space>
    // expected-error @+1 {{cpu.alloc: unsupported use of the returned pointer}}
    func.call @use(%p) : (!ptr.ptr<#tptr.default_memory_space>) -> ()
    return
  }
}
