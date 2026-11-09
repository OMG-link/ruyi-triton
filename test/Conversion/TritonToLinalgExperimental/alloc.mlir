// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s
// RUN: triton-shared-opt --triton-to-linalg-experimental %s | mlir-opt --one-shot-bufferize --buffer-deallocation-pipeline | FileCheck %s --check-prefix=DEALLOC
//
// tt.dev.alloc with the default placement becomes a heap memref.alloc, and loads and stores through
// the returned pointer become structured memref accesses on that allocation,
// the same as through a pointer argument. The ownership-based deallocation
// pipeline frees the allocation before the function returns.

module {
  // CHECK-LABEL: func.func @alloc_roundtrip
  // DEALLOC-LABEL: func.func @alloc_roundtrip
  tt.func public @alloc_roundtrip(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>, %off: i32) {
    // CHECK: %[[ALLOC:.+]] = memref.alloc() alignment = 64 : memref<256xf32>
    // CHECK-NOT: tt.dev.alloc
    // CHECK-NOT: unrealized_conversion_cast
    %buf = tt.dev.alloc {alignment = 64 : i32, num_elements = 256 : i64} : !tt.ptr<f32>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %in_s = tt.splat %in : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %in_p = tt.addptr %in_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %x = tt.load %in_p : tensor<64x!tt.ptr<f32>>

    // Store to scratch at a dynamic offset.
    // CHECK: %[[VIEW:.+]] = memref.reinterpret_cast %[[ALLOC]] to offset: [%{{[a-z0-9_]+}}], sizes: [64], strides: [1]
    // CHECK: bufferization.materialize_in_destination %{{.*}} in writable %[[VIEW]]
    %buf_off = tt.addptr %buf, %off : !tt.ptr<f32>, i32
    %buf_s = tt.splat %buf_off : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %buf_p = tt.addptr %buf_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %buf_p, %x : tensor<64x!tt.ptr<f32>>

    // Read it back from the same place.
    // CHECK: memref.copy %[[VIEW]]
    %y = tt.load %buf_p : tensor<64x!tt.ptr<f32>>
    %out_s = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_p = tt.addptr %out_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_p, %y : tensor<64x!tt.ptr<f32>>

    // DEALLOC: %[[A:.+]] = memref.alloc() alignment = 64 : memref<256xf32>
    // DEALLOC: memref.dealloc %[[A]]
    // DEALLOC: return
    // The conversion itself leaves freeing to the deallocation pipeline.
    // CHECK-NOT: memref.dealloc
    // CHECK: return
    tt.return
  }
}
