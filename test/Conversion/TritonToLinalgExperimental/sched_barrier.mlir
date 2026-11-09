// RUN: triton-shared-opt --triton-to-linalg-experimental %s | FileCheck %s

module {
  // CHECK-LABEL: func.func @value_only
  // CHECK: cpu.sched_barrier %arg0 : tensor<16xf32>
  tt.func public @value_only(%arg0: tensor<16xf32>) {
    %r = tt.dev.sched_barrier %arg0 : (tensor<16xf32>) -> tensor<16xf32>
    tt.return
  }

  // CHECK-LABEL: func.func @with_memory
  // CHECK: cpu.sched_barrier %arg0 : tensor<16xf32> <memory>
  tt.func public @with_memory(%arg0: tensor<16xf32>) {
    %r = tt.dev.sched_barrier %arg0 {memory} : (tensor<16xf32>) -> tensor<16xf32>
    tt.return
  }
}
