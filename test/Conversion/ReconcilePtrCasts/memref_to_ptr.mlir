// RUN: triton-shared-opt %s --reconcile-ptr-casts | FileCheck %s

// Address extraction must not assume a particular runtime rank.
func.func @unknown_rank(%base: memref<*xf32>) -> !ptr.ptr<#tptr.default_memory_space> {
  %pointer = builtin.unrealized_conversion_cast %base : memref<*xf32> to !ptr.ptr<#tptr.default_memory_space>
  return %pointer : !ptr.ptr<#tptr.default_memory_space>
}

// CHECK-LABEL: func.func @unknown_rank(
// CHECK-NOT: memref.cast
// CHECK: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %arg0 : memref<*xf32> -> index
// CHECK: %[[POINTER:.*]] = tptr.inttoptr %[[ADDRESS]] : index to <#tptr.default_memory_space>
// CHECK: return %[[POINTER]]

func.func @rank_zero(%base: memref<f32>) -> !ptr.ptr<#tptr.default_memory_space> {
  %unranked = memref.cast %base : memref<f32> to memref<*xf32>
  %pointer = builtin.unrealized_conversion_cast %unranked : memref<*xf32> to !ptr.ptr<#tptr.default_memory_space>
  return %pointer : !ptr.ptr<#tptr.default_memory_space>
}

// CHECK-LABEL: func.func @rank_zero(
// CHECK: %[[BASE:.*]] = memref.cast %arg0 : memref<f32> to memref<*xf32>
// CHECK-NOT: memref<1xf32>
// CHECK: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]] : memref<*xf32> -> index
// CHECK: tptr.inttoptr %[[ADDRESS]] : index to <#tptr.default_memory_space>

// This conversion recovers a storage base, not the first logical element of a
// view. Keep the view offset separate; it must not be added a second time by
// consumers that already carry their access offsets in the pointer IR.
func.func @storage_base(%view: memref<4xf32, strided<[2], offset: 7>>) -> !ptr.ptr<#tptr.default_memory_space> {
  %unranked = memref.cast %view : memref<4xf32, strided<[2], offset: 7>> to memref<*xf32>
  %pointer = builtin.unrealized_conversion_cast %unranked : memref<*xf32> to !ptr.ptr<#tptr.default_memory_space>
  return %pointer : !ptr.ptr<#tptr.default_memory_space>
}

// CHECK-LABEL: func.func @storage_base(
// CHECK: %[[BASE:.*]] = memref.cast %arg0 : memref<4xf32, strided<[2], offset: 7>> to memref<*xf32>
// CHECK: %[[ADDRESS:.*]] = memref.extract_aligned_pointer_as_index %[[BASE]] : memref<*xf32> -> index
// CHECK-NEXT: %[[POINTER:.*]] = tptr.inttoptr %[[ADDRESS]] : index to <#tptr.default_memory_space>
// CHECK-NEXT: return %[[POINTER]]
