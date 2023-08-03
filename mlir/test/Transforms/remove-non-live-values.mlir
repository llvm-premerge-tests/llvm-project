// RUN: mlir-opt %s -remove-non-live-values -split-input-file -verify-diagnostics | FileCheck %s

// CHECK-LABEL: module @dont_touch_unacceptable_ir {
// CHECK-LABEL:   func.func @has_cleanable_simple_op(%arg0: i32) {
// CHECK-NEXT:      %0 = arith.addi %arg0, %arg0 : i32
// CHECK-NEXT:      return
// CHECK-NEXT:    }
// CHECK-NEXT:  }
//
// The IR remains untouched because of the presence of a non-function-like
// symbol op (module @dont_touch_unacceptable_ir).
// expected-warning @+1 {{Unacceptable IR encountered for the optimization pass `remove-non-live-values`. Pass won't execute.}}
module @dont_touch_unacceptable_ir {
  func.func @has_cleanable_simple_op(%arg0 : i32) {
    %0 = arith.addi %arg0, %arg0 : i32
    return
  }
}

// -----

// CHECK-LABEL: func.func @dont_touch_unacceptable_ir_has_cleanable_simple_op_with_branch_op(%arg0: i1) {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    cf.cond_br %arg0, ^bb1(%c0_i32 : i32), ^bb2(%c0_i32 : i32)
// CHECK-NEXT:  ^bb1(%0: i32):
// CHECK-NEXT:    cf.br ^bb3
// CHECK-NEXT:  ^bb2(%1: i32):
// CHECK-NEXT:    cf.br ^bb3
// CHECK-NEXT:  ^bb3:
// CHECK-NEXT:    return
// CHECK-NEXT:  }
//
// The IR remains untouched because of the presence of a branch op `cf.cond_br`.
func.func @dont_touch_unacceptable_ir_has_cleanable_simple_op_with_branch_op(%arg0: i1) {
  %c0_i32 = arith.constant 0 : i32
  // expected-warning @+1 {{Unacceptable IR encountered for the optimization pass `remove-non-live-values`. Pass won't execute.}}
  cf.cond_br %arg0, ^bb1(%c0_i32 : i32), ^bb2(%c0_i32 : i32)
^bb1(%0 : i32):
  cf.br ^bb3
^bb2(%1 : i32):
  cf.br ^bb3
^bb3:
  return
}

// -----

// CHECK-LABEL: func.func @clean_simple_op(%arg0: i32) {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
//
// arith.addi (a simple op) gets removed but %arg0 doesn't get removed from the
// function because the signature of a public function is always left untouched.
func.func @clean_simple_op(%arg0 : i32) {
  %0 = arith.addi %arg0, %arg0 : i32
  return
}

// -----

// CHECK-LABEL: func.func private @f() {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
// CHECK-LABEL: func.func @clean_func_op_remove_argument_and_return_value(%arg0: i32) {
// CHECK-NEXT:    call @f() : () -> ()
// CHECK-NEXT:    return
// CHECK-NEXT:  }
func.func private @f(%arg0: i32) -> (i32) {
  return %arg0 : i32
}
func.func @clean_func_op_remove_argument_and_return_value(%arg0 : i32) {
  %0 = func.call @f(%arg0) : (i32) -> (i32)
  return
}

// -----

// CHECK-LABEL: func.func private @f() -> i32 {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    return %c0_i32 : i32
// CHECK-NEXT:  }
// CHECK-LABEL: func.func @clean_func_op_remove_arguments(%arg0: memref<i32>, %arg1: i32) -> (i32, memref<i32>) {
// CHECK-NEXT:    %0 = call @f() : () -> i32
// CHECK-NEXT:    return %0, %arg0 : i32, memref<i32>
// CHECK-NEXT:  }
//
// %arg0 is not live in @f because it is never used. %arg1 is not live in @f
// because its user `arith.addi` doesn't have any uses and the %0#1 value that
// it is forwarded to also doesn't have any uses.
func.func private @f(%arg0 : memref<i32>, %arg1 : i32) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  %0 = arith.addi %arg1, %arg1 : i32
  return %c0_i32, %arg1 : i32, i32
}
func.func @clean_func_op_remove_arguments(%arg0 : memref<i32>, %arg1 : i32) -> (i32, memref<i32>) {
  %0:2 = func.call @f(%arg0, %arg1) : (memref<i32>, i32) -> (i32, i32)
  return %0#0, %arg0 : i32, memref<i32>
}

// -----

// CHECK-LABEL: func.func private @f(%arg0: memref<i32>) -> i32 {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    memref.store %c0_i32, %arg0[] : memref<i32>
// CHECK-NEXT:    return %c0_i32 : i32
// CHECK-NEXT:  }
// CHECK-LABEL: func.func @clean_func_op_remove_return_values(%arg0: memref<i32>) -> i32 {
// CHECK-NEXT:    %0 = call @f(%arg0) : (memref<i32>) -> i32
// CHECK-NEXT:    %1 = call @f(%arg0) : (memref<i32>) -> i32
// CHECK-NEXT:    return %0 : i32
// CHECK-NEXT:  }
//
// Even though %4#0 was not live, the first return value of @f isn't removed
// because %0#0 was live (liveness is checked across all callers).
//
// Also, the second return value of @f is removed despite %c0_i32 being live
// because neither %0#1 nor %4#1 were live (removal doesn't depend on the
// liveness of the operand itself but on the liveness of where it is forwarded).
func.func private @f(%arg0 : memref<i32>) -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  memref.store %c0_i32, %arg0[] : memref<i32>
  return %c0_i32, %c0_i32 : i32, i32
}
func.func @clean_func_op_remove_return_values(%arg0 : memref<i32>) -> (i32) {
  %0:2 = func.call @f(%arg0) : (memref<i32>) -> (i32, i32)
  %1 = arith.addi %0#0, %0#1 : i32
  %2 = arith.addi %1, %0#0 : i32
  %3 = arith.muli %2, %0#1 : i32
  %4:2 = func.call @f(%arg0) : (memref<i32>) -> (i32, i32)
  return %0#0 : i32
}

// -----

// CHECK-LABEL: func.func private @f() -> (i32, i32) {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    return %c0_i32, %c0_i32 : i32, i32
// CHECK-NEXT:  }
// CHECK-LABEL: func.func @clean_func_op_dont_remove_return_values() -> (i32, i32) {
// CHECK-NEXT:    %0:2 = call @f() : () -> (i32, i32)
// CHECK-NEXT:    %1:2 = call @f() : () -> (i32, i32)
// CHECK-NEXT:    return %0#0, %1#1 : i32, i32
// CHECK-NEXT:  }
//
// None of the return values of @f can be removed because the first one is
// forwarded to a live value %0#0 and the second one is forwarded to a live
// value %1#1.
func.func private @f() -> (i32, i32) {
  %c0_i32 = arith.constant 0 : i32
  return %c0_i32, %c0_i32 : i32, i32
}
func.func @clean_func_op_dont_remove_return_values() -> (i32, i32) {
  %0:2 = func.call @f() : () -> (i32, i32)
  %1:2 = func.call @f() : () -> (i32, i32)
  return %0#0, %1#1 : i32, i32
}

// -----

// CHECK-LABEL: func.func @clean_region_branch_op_erase_it(%arg0: i32, %arg1: i1) {
// CHECK-NEXT:    return
// CHECK-NEXT:  }
//
// The scf.while op has no memory effects and none of its results are live.
func.func @clean_region_branch_op_erase_it(%arg0 : i32, %arg1 : i1) {
  %0 = scf.while (%arg2 = %arg0) : (i32) -> (i32) {
    %1 = arith.muli %arg2, %arg2 : i32
    scf.condition(%arg1) %arg2 : i32
  } do {
  ^bb0(%arg2: i32):
    %2 = arith.muli %arg2, %arg2 : i32
    scf.yield %2 : i32
  }
  %3 = arith.addi %0, %0 : i32
  %4 = arith.muli %0, %3 : i32
  return
}

// -----

// CHECK-LABEL: func.func @clean_region_branch_op_keep_results_and_second_region_arg_and_remove_first_region_arg(%arg0: i1) -> i32 {
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %0:2 = scf.while (%arg1 = %c1_i32) : (i32) -> (i32, i32) {
// CHECK-NEXT:      scf.condition(%arg0) %arg1, %arg1 : i32, i32
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%arg1: i32, %arg2: i32):
// CHECK-NEXT:      scf.yield %arg2 : i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#0 : i32
// CHECK-NEXT:  }
//
// Values kept:
//  (1) %0#1 is not live. Yet, it is kept because %arg2 (the second instance in
//  `scf.condition`) forwards to it and this instance of %arg2 has to be kept.
//  The second instance of %arg2 in `scf.condition` has to be kept because it
//  forwards to %arg4 which is live.
//
//  (2) %arg3 is not live. Yet, it is kept because %arg2 (the first instance)
//  forwards to it and this instance of %arg2 has to be kept.
//  The first instance of %arg2 in `scf.condition` has to be kept because it
//  forwards to %0#0 which is live.
//
// Values not kept:
//  (1) %c0_i32 is not kept as an operand of `scf.while` because it only
//  forwards to %arg1, which is not kept. %arg1 is not kept because only %c0_i32
//  and %arg4 forward to it, neither of them forward anywhere else and %arg1 is
//  not live. Thus, %arg4 is also not kept (the first instance) in the
//  `scf.yield` op.
func.func @clean_region_branch_op_keep_results_and_second_region_arg_and_remove_first_region_arg(%arg0: i1) -> (i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0:2 = scf.while (%arg1 = %c0_i32, %arg2 = %c1_i32) : (i32, i32) -> (i32, i32) {
    %1 = arith.addi %arg1, %arg2 : i32
    scf.condition(%arg0) %arg2, %arg2 : i32, i32
  } do {
  ^bb0(%arg3: i32, %arg4: i32):
    %2 = arith.addi %arg3, %arg3 : i32
    scf.yield %arg4, %arg4 : i32, i32
  }
  return %0#0 : i32
}

// -----

// CHECK-LABEL: func.func @clean_region_branch_op_remove_results_and_second_region_arg_and_keep_first_region_arg(%arg0: i1) -> i32 {
// CHECK-NEXT:    %c0_i32 = arith.constant 0 : i32
// CHECK-NEXT:    %c1_i32 = arith.constant 1 : i32
// CHECK-NEXT:    %0:2 = scf.while (%arg1 = %c0_i32, %arg2 = %c1_i32) : (i32, i32) -> (i32, i32) {
// CHECK-NEXT:      scf.condition(%arg0) %arg2, %arg1 : i32, i32
// CHECK-NEXT:    } do {
// CHECK-NEXT:    ^bb0(%arg1: i32, %arg2: i32):
// CHECK-NEXT:      scf.yield %arg1, %arg2 : i32, i32
// CHECK-NEXT:    }
// CHECK-NEXT:    return %0#0 : i32
// CHECK-NEXT:  }
//
// Values kept:
//  (1) %0#0 is kept because it is live.
//
//  (2) %0#1 is not live. Yet, it is kept because %arg1 (the first instance in
//  `scf.condition`) forwards to it and this instance of %arg1 has to be kept.
//  The first instance of %arg1 in `scf.condition` has to be kept because it
//  forwards to %arg4, which forwards to %arg2, which forwards to %0#0, which is
//  live.
//
// Values not kept:
//  (1) %0#2 is not kept because the second instance of %arg1 in `scf.condition`
//  forwards to it, which forwards to only %0#2 and %arg5, where both these are
//  not live and have no other value forwarding to them.
//
//  (2) %0#3 is not kept because the third instance of %arg1 in `scf.condition`
//  forwards to it, which forwards to only %0#3 and %arg6, where both these are
//  not live and have no other value forwarding to them.
func.func @clean_region_branch_op_remove_results_and_second_region_arg_and_keep_first_region_arg(%arg0: i1) -> (i32) {
  %c0_i32 = arith.constant 0 : i32
  %c1_i32 = arith.constant 1 : i32
  %0:4 = scf.while (%arg1 = %c0_i32, %arg2 = %c1_i32) : (i32, i32) -> (i32, i32, i32, i32) {
    scf.condition(%arg0) %arg2, %arg1, %arg1, %arg1 : i32, i32, i32, i32
  } do {
  ^bb0(%arg3: i32, %arg4: i32, %arg5: i32, %arg6: i32):
    scf.yield %arg3, %arg4 : i32, i32
  }
  return %0#0 : i32
}

// -----

// CHECK-LABEL: func.func @clean_region_branch_op_remove_result(%arg0: index, %arg1: memref<i32>) {
// CHECK-NEXT:    scf.index_switch %arg0
// CHECK-NEXT:    case 1 {
// CHECK-NEXT:      %c10_i32 = arith.constant 10 : i32
// CHECK-NEXT:      memref.store %c10_i32, %arg1[] : memref<i32>
// CHECK-NEXT:      scf.yield
// CHECK-NEXT:    }
// CHECK-NEXT:    default {
// CHECK-NEXT:    }
// CHECK-NEXT:    return
// CHECK-NEXT:  }
//
// The op isn't erased because it has memory effects but its unnecessary result
// is removed.
func.func @clean_region_branch_op_remove_result(%cond : index, %arg0 : memref<i32>) {
  %1 = scf.index_switch %cond -> i32
  case 1 {
    %c10_i32 = arith.constant 10 : i32
    memref.store %c10_i32, %arg0[] : memref<i32>
    scf.yield %c10_i32 : i32
  }
  default {
    %c11_i32 = arith.constant 11 : i32
    scf.yield %c11_i32 : i32
  }
  return
}
