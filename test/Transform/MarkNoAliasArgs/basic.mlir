// RUN: triton-shared-opt --mark-noalias-args %s | FileCheck %s
// RUN: triton-shared-opt --mark-noalias-args --mark-noalias-args %s | FileCheck %s
// RUN: triton-shared-opt --mark-noalias-args %s | triton-shared-opt --mark-noalias-args | FileCheck %s
//
// --mark-noalias-args sets llvm.noalias on every !tt.ptr and memref
// argument, leaves scalar arguments unmarked, keeps existing argument
// attributes, and is idempotent (the second and third RUN lines apply it
// twice and expect the same output).

// Pointer arguments are marked; scalar arguments are not.
// CHECK-LABEL: tt.func @ptr_and_scalar
// CHECK-SAME: (%{{[a-z0-9_]+}}: !tt.ptr<f32> {llvm.noalias},
// CHECK-SAME: %{{[a-z0-9_]+}}: !tt.ptr<i8> {llvm.noalias},
// CHECK-SAME: %{{[a-z0-9_]+}}: i32)
tt.func @ptr_and_scalar(%a: !tt.ptr<f32>, %b: !tt.ptr<i8>, %n: i32) {
  tt.return
}

// Existing attributes are kept and an existing mark is not duplicated.
// CHECK-LABEL: tt.func @existing_attrs
// CHECK-SAME: (%{{[a-z0-9_]+}}: !tt.ptr<f32> {llvm.noalias, tt.divisibility = 16 : i32},
// CHECK-SAME: %{{[a-z0-9_]+}}: !tt.ptr<f32> {llvm.noalias},
// CHECK-SAME: %{{[a-z0-9_]+}}: i32 {tt.divisibility = 16 : i32})
tt.func @existing_attrs(%a: !tt.ptr<f32> {tt.divisibility = 16 : i32},
                        %b: !tt.ptr<f32> {llvm.noalias},
                        %n: i32 {tt.divisibility = 16 : i32}) {
  tt.return
}

// Memref arguments of func.func are marked too.
// CHECK-LABEL: func.func @memref_args
// CHECK-SAME: (%{{[a-z0-9_]+}}: memref<*xf32> {llvm.noalias},
// CHECK-SAME: %{{[a-z0-9_]+}}: memref<4xi8> {llvm.noalias},
// CHECK-SAME: %{{[a-z0-9_]+}}: index)
func.func @memref_args(%a: memref<*xf32>, %b: memref<4xi8>, %i: index) {
  return
}

// CHECK-NOT: llvm.noalias, llvm.noalias
