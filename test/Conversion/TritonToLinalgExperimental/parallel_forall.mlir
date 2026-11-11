// RUN: triton-shared-opt --triton-to-linalg-experimental=structured-ldst-mode=tensor-first-vector-cpu %s | FileCheck %s
//
// A tl.dev.parallel loop is a 1-d scf.forall without shared outputs. It passes
// through unchanged, and the Triton ops of its body are converted like those
// of any other region: here a structured load and store.

// CHECK-LABEL: func.func @parallel_forall
// CHECK: %[[UB:.*]] = arith.index_cast %{{.*}} : i32 to index
// CHECK: scf.forall (%[[IV:.*]]) = (0) to (%[[UB]]) step (64) {
// CHECK: arith.index_cast %[[IV]] : index to i32
// CHECK: memref.reinterpret_cast %{{.*}} to offset: [%{{.*}}], sizes: [64], strides: [1]
// CHECK: bufferization.to_tensor
// CHECK: linalg.generic
// CHECK: bufferization.materialize_in_destination
// CHECK-NEXT: }
// CHECK-NEXT: return
// CHECK-NOT: tt.
module {
  tt.func public @parallel_forall(%x: !tt.ptr<f32>, %out: !tt.ptr<f32>, %n: i32) {
    %c0 = arith.constant 0 : i32
    %c64 = arith.constant 64 : i32
    %pid = tt.get_program_id x : i32
    %base = arith.muli %pid, %n : i32
    %lb = arith.index_cast %c0 : i32 to index
    %ub = arith.index_cast %n : i32 to index
    %st = arith.index_cast %c64 : i32 to index
    scf.forall (%iv) = (%lb) to (%ub) step (%st) {
      %i = arith.index_cast %iv : index to i32
      %off = arith.addi %base, %i : i32
      %r = tt.make_range {end = 64 : i32, start = 0 : i32} : tensor<64xi32>
      %s = tt.splat %off : i32 -> tensor<64xi32>
      %o = arith.addi %s, %r : tensor<64xi32>
      %px = tt.splat %x : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %ax = tt.addptr %px, %o : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      %v = tt.load %ax : tensor<64x!tt.ptr<f32>>
      %v2 = arith.addf %v, %v : tensor<64xf32>
      %po = tt.splat %out : !tt.ptr<f32> -> tensor<64x!tt.ptr<f32>>
      %ao = tt.addptr %po, %o : tensor<64x!tt.ptr<f32>>, tensor<64xi32>
      tt.store %ao, %v2 : tensor<64x!tt.ptr<f32>>
    }
    tt.return
  }
}
