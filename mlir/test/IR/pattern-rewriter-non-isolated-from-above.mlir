// RUN: mlir-opt --test-pattern-non-isolated-from-above %s | FileCheck %s

func.func @test(%arg0 : i32) {
  %0 = "test.cast"(%arg0) : (i32) -> f32
  "test.one_region_op"()({
    %1 = "test.cast"(%0) : (f32) -> i32
    "test.region_yield"(%1) : (i32) -> ()
  }) : () -> ()
  return
}
// CHECK-LABEL: func @test(
//  CHECK-SAME:     %[[ARG0:.+]]: i32)
//       CHECK:   test.one_region_op
//       CHECK:     test.region_yield
//  CHECK-SAME:         %[[ARG0]]
