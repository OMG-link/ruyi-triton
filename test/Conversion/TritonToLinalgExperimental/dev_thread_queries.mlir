// RUN: triton-shared-opt --triton-to-linalg-experimental=structured-ldst-mode=tensor-first-vector-cpu %s | FileCheck %s
//
// tt.dev.current_cpu and tt.dev.os_thread_id become calls of the CPU
// runtime's functions. Every query stays a call of its own: two queries
// inside a parallel loop are neither merged nor hoisted out of it.

// CHECK-DAG: func.func private @tcrt_current_cpu() -> i32
// CHECK-DAG: func.func private @tcrt_os_thread_id() -> i32
// CHECK-LABEL: func.func @thread_queries
// CHECK-NOT: tt.dev
// CHECK: scf.forall
// CHECK: call @tcrt_current_cpu() : () -> i32
// CHECK: call @tcrt_current_cpu() : () -> i32
// CHECK: call @tcrt_os_thread_id() : () -> i32
// CHECK: }
// CHECK-NOT: tt.dev
module {
  tt.func public @thread_queries(%out: !tt.ptr<i32>, %n: i32) {
    %lb = arith.constant 0 : index
    %ub = arith.index_cast %n : i32 to index
    %st = arith.constant 1 : index
    scf.forall (%iv) = (%lb) to (%ub) step (%st) {
      %i = arith.index_cast %iv : index to i32
      %c0 = tt.dev.current_cpu : i32
      %c1 = tt.dev.current_cpu : i32
      %t = tt.dev.os_thread_id : i32
      %s = arith.addi %c0, %c1 : i32
      %v = arith.addi %s, %t : i32
      %p = tt.addptr %out, %i : !tt.ptr<i32>, i32
      tt.store %p, %v : !tt.ptr<i32>
    }
    tt.return
  }
}
