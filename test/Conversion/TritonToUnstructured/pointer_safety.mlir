// RUN: triton-shared-opt --split-input-file --triton-to-unstructured --verify-each %s 2>&1 | FileCheck %s

module {
  tt.func public @byte_sized_tensor_bitcast(%base: !tt.ptr<i1>, %offsets: tensor<4xi32>) -> tensor<4xi8> {
    %ptrs = tt.splat %base : !tt.ptr<i1> -> tensor<4x!tt.ptr<i1>>
    %shifted = tt.addptr %ptrs, %offsets : tensor<4x!tt.ptr<i1>>, tensor<4xi32>
    %cast = tt.bitcast %shifted : tensor<4x!tt.ptr<i1>> -> tensor<4x!tt.ptr<i8>>
    %value = tt.load %cast : tensor<4x!tt.ptr<i8>>
    tt.return %value : tensor<4xi8>
  }
}

// CHECK-LABEL: tt.func public @byte_sized_tensor_bitcast(
// CHECK-SAME:    %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i1>, %[[OFFSETS:.*]]: tensor<4xi32>)
// CHECK:         %[[INDEX:.*]] = arith.index_cast %[[OFFSETS]] : tensor<4xi32> to tensor<4xindex>
// CHECK:         %[[CAST:.*]] = tt.bitcast %[[BASE]] : !tt.ptr<i1> -> !tt.ptr<i8>
// CHECK:         %[[VALUE:.*]] = tts.gather %[[CAST]][%[[INDEX]]] : (<i8>, tensor<4xindex>) -> tensor<4xi8>
// CHECK:         tt.return %[[VALUE]] : tensor<4xi8>

// -----

module {
  tt.func public @while_changes_pointer_base(%base0: !tt.ptr<i32>, %base1: !tt.ptr<i32>) -> (i32, i1) {
    %true = arith.constant true
    %false = arith.constant false
    %c1 = arith.constant 1 : i32
    %result:2 = scf.while (%ptr = %base0, %keep_going = %true) : (!tt.ptr<i32>, i1) -> (!tt.ptr<i32>, i1) {
      scf.condition(%keep_going) %ptr, %keep_going : !tt.ptr<i32>, i1
    } do {
    ^bb0(%ptr: !tt.ptr<i32>, %keep_going: i1):
      %next = tt.addptr %base1, %c1 : !tt.ptr<i32>, i32
      scf.yield %next, %false : !tt.ptr<i32>, i1
    }
    %value = tt.load %result#0 : !tt.ptr<i32>
    tt.return %value, %result#1 : i32, i1
  }
}

// CHECK-LABEL: tt.func public @while_changes_pointer_base(
// CHECK-SAME:    %[[BASE0:.*]]: !tt.ptr<i32>, %[[BASE1:.*]]: !tt.ptr<i32>)
// CHECK:         %[[RESULT:.*]]:2 = scf.while (%[[PTR:.*]] = %[[BASE0]]
// CHECK:           scf.condition(%{{.*}}) %[[PTR]], %{{.*}} : !tt.ptr<i32>, i1
// CHECK:         } do {
// CHECK:         ^bb0(%{{.*}}: !tt.ptr<i32>, %{{.*}}: i1):
// CHECK:           %[[NEXT:.*]] = tt.addptr %[[BASE1]]
// CHECK:           scf.yield %[[NEXT]], %{{.*}} : !tt.ptr<i32>, i1
// CHECK:         }
// CHECK:         %[[VALUE:.*]] = tt.load %[[RESULT]]#0 : !tt.ptr<i32>
// CHECK:         tt.return %[[VALUE]], %[[RESULT]]#1 : i32, i1

// -----

module {
  tt.func public @for_direct_base(%base: !tt.ptr<i32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %result = scf.for %iv = %c0 to %c2 step %c1 iter_args(%ptr = %base) -> (!tt.ptr<i32>) {
      scf.yield %base : !tt.ptr<i32>
    }
    %value = tt.load %result : !tt.ptr<i32>
    tt.return %value : i32
  }
}

// CHECK-LABEL: tt.func public @for_direct_base(
// CHECK-SAME:    %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
// CHECK-NOT:     scf.for
// CHECK:         %[[VALUE:.*]] = tts.gather %[[BASE]][%[[ZERO]]] : (<i32>, index) -> i32
// CHECK:         tt.return %[[VALUE]] : i32

// -----

module {
  tt.func public @for_widens_offset(%base: !tt.ptr<i32>) -> i32 {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %c1_i64 = arith.constant 1 : i64
    %result = scf.for %iv = %c0 to %c2 step %c1 iter_args(%ptr = %base) -> (!tt.ptr<i32>) {
      %next = tt.addptr %ptr, %c1_i64 : !tt.ptr<i32>, i64
      scf.yield %next : !tt.ptr<i32>
    }
    %value = tt.load %result : !tt.ptr<i32>
    tt.return %value : i32
  }
}

// CHECK-LABEL: tt.func public @for_widens_offset(
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:         %[[ONE:.*]] = arith.constant 1 : index
// CHECK:         %[[RESULT:.*]] = scf.for {{.*}} iter_args(%[[PTR:.*]] = %[[ZERO]]) -> (index)
// CHECK:           %[[NEXT:.*]] = arith.addi %[[PTR]], %[[ONE]] : index
// CHECK:           scf.yield %[[NEXT]] : index
// CHECK:         %[[VALUE:.*]] = tts.gather %arg0[%[[RESULT]]] : (<i32>, index) -> i32
// CHECK:         tt.return %[[VALUE]] : i32

// -----

module {
  tt.func public @while_keeps_pointer_base(%base: !tt.ptr<i32>) -> (i32, i1) {
    %true = arith.constant true
    %false = arith.constant false
    %result:2 = scf.while (%ptr = %base, %keep_going = %true) : (!tt.ptr<i32>, i1) -> (!tt.ptr<i32>, i1) {
      scf.condition(%keep_going) %base, %keep_going : !tt.ptr<i32>, i1
    } do {
    ^bb0(%ptr: !tt.ptr<i32>, %keep_going: i1):
      scf.yield %base, %false : !tt.ptr<i32>, i1
    }
    %value = tt.load %result#0 : !tt.ptr<i32>
    tt.return %value, %result#1 : i32, i1
  }
}

// CHECK-LABEL: tt.func public @while_keeps_pointer_base(
// CHECK-SAME:    %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i32>)
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:         %[[RESULT:.*]] = scf.while (%{{.*}} = %{{.*}}) : (i1) -> i1
// CHECK:           scf.condition(%{{.*}}) %{{.*}} : i1
// CHECK:           scf.yield %{{.*}} : i1
// CHECK:         %[[VALUE:.*]] = tts.gather %[[BASE]][%[[ZERO]]] : (<i32>, index) -> i32
// CHECK:         tt.return %[[VALUE]], %[[RESULT]] : i32, i1

// -----

module {
  tt.func public @wide_base_offset(%base: !tt.ptr<f32>, %base_offset: i64, %offsets: tensor<4xi32>) -> tensor<4xf32> {
    %shifted = tt.addptr %base, %base_offset : !tt.ptr<f32>, i64
    %ptr = tts.make_gather_scatter_tptr %shifted to sizes: [4] gather_scatter_dim: 0 gather_scatter_offset: %offsets, strides: [1], offsets: [0] : tensor<4xi32> <f32> to tensor<4x!tt.ptr<f32>>
    %value = "tts.load"(%ptr) <{operandSegmentSizes = array<i32: 1, 0, 0>, static_mask_dims = array<i64>}> : (tensor<4x!tt.ptr<f32>>) -> tensor<4xf32>
    tt.return %value : tensor<4xf32>
  }
}

// CHECK-LABEL: tt.func public @wide_base_offset(
// CHECK-NOT:     arith.trunci
// CHECK:         %[[BASE_OFFSET:.*]] = arith.index_cast %arg1 : i64 to index
// CHECK:         %[[OFFSETS:.*]] = arith.index_cast %arg2 : tensor<4xi32> to tensor<4xindex>
// CHECK:         %[[BASE_SPLAT:.*]] = tt.splat %[[BASE_OFFSET]] : index -> tensor<4xindex>
// CHECK:         %[[ACCUMULATED:.*]] = arith.addi %[[BASE_SPLAT]], %[[OFFSETS]] : tensor<4xindex>
// CHECK-NOT:     tt.addptr
// CHECK:         tts.make_gather_scatter_tptr %arg0 {{.*}} gather_scatter_offset: %[[ACCUMULATED]]{{.*}} : tensor<4xindex> <f32> to tensor<4x!tt.ptr<f32>>

// -----

module {
  tt.func public @scalar_bitcast_multiple_users(%base: !tt.ptr<i32>, %offset: i32) -> (i32, i64) {
    %shifted = tt.addptr %base, %offset : !tt.ptr<i32>, i32
    %first = tt.load %shifted : !tt.ptr<i32>
    %wide = tt.bitcast %shifted : !tt.ptr<i32> -> !tt.ptr<i64>
    %second = tt.load %wide : !tt.ptr<i64>
    tt.return %first, %second : i32, i64
  }
}

// CHECK-LABEL: tt.func public @scalar_bitcast_multiple_users(
// CHECK-SAME:    %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i32>, %[[OFFSET:.*]]: i32)
// CHECK:         %[[ZERO:.*]] = arith.constant 0 : index
// CHECK:         %[[INDEX:.*]] = arith.index_cast %[[OFFSET]] : i32 to index
// CHECK:         %[[FIRST:.*]] = tts.gather %[[BASE]][%[[INDEX]]] : (<i32>, index) -> i32
// CHECK:         %[[ADDRESS:.*]] = tt.addptr %[[BASE]], %[[INDEX]] : !tt.ptr<i32>, index
// CHECK:         %[[WIDE:.*]] = tt.bitcast %[[ADDRESS]] : !tt.ptr<i32> -> !tt.ptr<i64>
// CHECK:         %[[SECOND:.*]] = tts.gather %[[WIDE]][%[[ZERO]]] : (<i64>, index) -> i64
// CHECK:         tt.return %[[FIRST]], %[[SECOND]] : i32, i64

// -----

module {
  tt.func public @same_width_bitcast_ptr_to_int(%base: !tt.ptr<i32>, %offset: i32) -> i64 {
    %shifted = tt.addptr %base, %offset : !tt.ptr<i32>, i32
    %cast = tt.bitcast %shifted : !tt.ptr<i32> -> !tt.ptr<f32>
    %address = tt.ptr_to_int %cast : !tt.ptr<f32> -> i64
    tt.return %address : i64
  }
}

// CHECK-LABEL: tt.func public @same_width_bitcast_ptr_to_int(
// CHECK-SAME:    %[[BASE:[a-zA-Z0-9_]+]]: !tt.ptr<i32>, %[[OFFSET:.*]]: i32)
// CHECK:         %[[INDEX:.*]] = arith.index_cast %[[OFFSET]] : i32 to index
// CHECK:         %[[CAST_BASE:.*]] = tt.bitcast %[[BASE]] : !tt.ptr<i32> -> !tt.ptr<f32>
// CHECK:         %[[ADDRESS:.*]] = tt.addptr %[[CAST_BASE]], %[[INDEX]] : !tt.ptr<f32>, index
// CHECK:         %[[RESULT:.*]] = tt.ptr_to_int %[[ADDRESS]] : !tt.ptr<f32> -> i64
// CHECK:         tt.return %[[RESULT]] : i64
