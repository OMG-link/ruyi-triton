// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental --verify-diagnostics %s | FileCheck %s
//
// tt.alloc honours its placement: global (the default) is a heap
// memref.alloc, local a stack memref.alloca, both with the requested
// alignment. spm has no lowering yet and is rejected rather than placed
// elsewhere.

module {
  // CHECK-LABEL: func.func @global_and_local
  tt.func public @global_and_local(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    // CHECK-DAG: memref.alloc() alignment = 64 : memref<64xf32>
    // CHECK-DAG: memref.alloca() alignment = 128 : memref<64xf32>
    // CHECK-NOT: tt.alloc
    %global = tt.alloc {alignment = 64 : i32, num_elements = 64 : i64, placement = 1 : i32} : !tt.ptr<f32>
    %local = tt.alloc {alignment = 128 : i32, num_elements = 64 : i64, placement = 2 : i32} : !tt.ptr<f32>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %in_s = tt.splat %in : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %in_p = tt.addptr %in_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %x = tt.load %in_p : tensor<64x!tt.ptr<f32>>
    %h_s = tt.splat %global : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %h_p = tt.addptr %h_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %h_p, %x : tensor<64x!tt.ptr<f32>>
    %y = tt.load %h_p : tensor<64x!tt.ptr<f32>>
    %s_s = tt.splat %local : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %s_p = tt.addptr %s_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %s_p, %y : tensor<64x!tt.ptr<f32>>
    %z = tt.load %s_p : tensor<64x!tt.ptr<f32>>
    %out_s = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_p = tt.addptr %out_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_p, %z : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  tt.func public @spm(%out: !tt.ptr<f32>) {
    // expected-error @+1 {{tt.alloc placement 'spm' is not implemented yet}}
    %buf = tt.alloc {alignment = 64 : i32, num_elements = 64 : i64, placement = 3 : i32} : !tt.ptr<f32>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %b_s = tt.splat %buf : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %b_p = tt.addptr %b_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %x = tt.load %b_p : tensor<64x!tt.ptr<f32>>
    %out_s = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_p = tt.addptr %out_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_p, %x : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}
