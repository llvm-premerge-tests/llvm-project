// A regression test for https://github.com/llvm/llvm-project/issues/64665 in
// which the dialect conversion framework tried to remove inserted
// unrealized_conversion_cast ops twice.

// RUN: mlir-opt %s -test-drop-dozing > %t
// RUN: FileCheck %s < %t

// CHECK-LABEL: example_fn
// CHECK-NOT: fall_asleep
func.func @example_fn(
    %s1 : !test.dozing<i32>,
    %s2 : !test.dozing<i32>) -> !test.dozing<i32> {
  func.return %s1 : !test.dozing<i32>
}

func.func @test_convert_call() -> i32 {
  %0 = arith.constant 7 : i32
  %1 = arith.constant 8 : i32
  %2 = test.fall_asleep %0 : i32 -> !test.dozing<i32>
  %3 = test.fall_asleep %1 : i32 -> !test.dozing<i32>
  %4 = func.call @example_fn(%2, %3) : (!test.dozing<i32>, !test.dozing<i32>) -> !test.dozing<i32>
  %5 = test.awaken %4 : !test.dozing<i32> -> i32
  func.return %5 : i32
}
