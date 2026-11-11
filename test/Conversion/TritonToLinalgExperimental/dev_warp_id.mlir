// RUN: triton-shared-opt --split-input-file --verify-diagnostics --triton-to-linalg-experimental=structured-ldst-mode=tensor-first-vector-cpu %s | FileCheck %s
//
// tt.dev.warp_id becomes a call of the CPU runtime's tcrt_warp_id, inside
// the parallel loop; outside of one it is rejected.

// CHECK: func.func private @tcrt_warp_id() -> i32
// CHECK-LABEL: func.func @warp_id
// CHECK: scf.forall
// CHECK: call @tcrt_warp_id() : () -> i32
// CHECK-NOT: tt.dev
module {
  tt.func public @warp_id(%out: !tt.ptr<i32>, %n: i32) {
    %lb = arith.constant 0 : index
    %ub = arith.index_cast %n : i32 to index
    %st = arith.constant 1 : index
    scf.forall (%iv) = (%lb) to (%ub) step (%st) {
      %i = arith.index_cast %iv : index to i32
      %w = tt.dev.warp_id : i32
      %p = tt.addptr %out, %i : !tt.ptr<i32>, i32
      tt.store %p, %w : !tt.ptr<i32>
    }
    tt.return
  }
}

// -----

module {
  tt.func public @outside(%out: !tt.ptr<i32>) {
    // expected-error @+1 {{is only valid inside a tl.dev.parallel loop}}
    %w = tt.dev.warp_id : i32
    tt.store %out, %w : !tt.ptr<i32>
    tt.return
  }
}
