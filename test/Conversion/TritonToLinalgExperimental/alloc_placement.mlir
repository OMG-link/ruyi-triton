// RUN: triton-shared-opt --split-input-file --triton-to-linalg-experimental %s | FileCheck %s
//
// tt.dev.alloc honours its placement: global (the default) is a heap
// memref.alloc, local a stack memref.alloca, both with the requested
// alignment. spm buffers are views of one hidden trailing argument, the
// scratchpad region the launcher provides, packed in program order at their
// alignments; the function records the bytes and alignment it needs.

module {
  // CHECK-LABEL: func.func @global_and_local
  tt.func public @global_and_local(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    // CHECK-DAG: memref.alloc() alignment = 64 : memref<64xf32>
    // CHECK-DAG: memref.alloca() alignment = 128 : memref<64xf32>
    // CHECK-NOT: tt.dev.alloc
    %global = tt.dev.alloc {alignment = 64 : i32, num_elements = 64 : i64, placement = 1 : i32} : !tt.ptr<f32>
    %local = tt.dev.alloc {alignment = 128 : i32, num_elements = 64 : i64, placement = 2 : i32} : !tt.ptr<f32>
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
  // The f32 buffer takes bytes [0, 256); the i8 one starts at its 128-byte
  // alignment, at 256, so the region is 272 bytes. The hidden argument comes
  // after the six program-id arguments.
  // CHECK-LABEL: func.func @spm(
  // CHECK-SAME: i32, %{{[a-z0-9_]+}}: i32, %[[SPM:[a-z0-9_]+]]: memref<272xi8>)
  // CHECK-SAME: triton_riscv.spm_align = 128 : i64, triton_riscv.spm_size = 272 : i64
  // CHECK-DAG: memref.view %[[SPM]][%c0{{[_0-9]*}}][] : memref<272xi8> to memref<64xf32>
  // CHECK-DAG: memref.view %[[SPM]][%c256{{[_0-9]*}}][] : memref<272xi8> to memref<16xi8>
  // CHECK-NOT: tt.dev.alloc
  tt.func public @spm(%in: !tt.ptr<f32>, %out: !tt.ptr<f32>) {
    %a = tt.dev.alloc {alignment = 64 : i32, num_elements = 64 : i64, placement = 3 : i32} : !tt.ptr<f32>
    %b = tt.dev.alloc {alignment = 128 : i32, num_elements = 16 : i64, placement = 3 : i32} : !tt.ptr<i8>
    %range = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
    %in_s = tt.splat %in : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %in_p = tt.addptr %in_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    %x = tt.load %in_p : tensor<64x!tt.ptr<f32>>
    %a_s = tt.splat %a : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %a_p = tt.addptr %a_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %a_p, %x : tensor<64x!tt.ptr<f32>>
    %y = tt.load %a_p : tensor<64x!tt.ptr<f32>>
    %r16 = tt.make_range {end = 16 : i32, start = 0 : i32} : tensor<16xi32>
    %b_s = tt.splat %b : !tt.ptr<i8> -> tensor<16x!tt.ptr<i8>>
    %b_p = tt.addptr %b_s, %r16 : tensor<16x!tt.ptr<i8>>, tensor<16xi32>
    %z = arith.constant dense<7> : tensor<16xi8>
    tt.store %b_p, %z : tensor<16x!tt.ptr<i8>>
    %out_s = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
    %out_p = tt.addptr %out_s, %range : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
    tt.store %out_p, %y : tensor<64x!tt.ptr<f32>>
    tt.return
  }
}

// -----

module {
  // A function without spm buffers gets no scratchpad argument.
  // CHECK-LABEL: func.func @no_spm(
  // CHECK-NOT: triton_riscv.spm_size
  // CHECK-SAME: {
  tt.func public @no_spm(%out: !tt.ptr<f32>) {
    %buf = tt.dev.alloc {alignment = 64 : i32, num_elements = 64 : i64, placement = 2 : i32} : !tt.ptr<f32>
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
