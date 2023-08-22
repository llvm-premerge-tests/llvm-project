// RUN: mlir-opt %s -commutative-operand-sort -cse | FileCheck %s

// CHECK-LABEL: func @check_commutative_cse
func.func @check_commutative_cse(%a : i32, %b : i32) -> i32 {
  %0 = arith.subi %a, %b : i32
  %1 = arith.divsi %a, %b : i32
  // CHECK: %[[ADD1:.*]] = arith.addi %{{.*}}, %{{.*}} : i32
  %2 = arith.addi %0, %1 : i32
  %3 = arith.addi %1, %0 : i32
  // CHECK-NEXT:  arith.muli %[[ADD1]], %[[ADD1]] : i32
  %4 = arith.muli %2, %3 : i32
  return %4 : i32
}
