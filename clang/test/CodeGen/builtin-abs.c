// RUN: %clang_cc1 -triple x86_64-unknown-unknown -emit-llvm %s -o - | FileCheck %s

int absi(int x) {
// CHECK-LABEL: @absi(
// CHECK:   [[ABS:%.*]] = call i32 @llvm.abs.i32(i32 [[X:%.*]], i1 true)
//
  return __builtin_abs(x);
}

long absl(long x) {
// CHECK-LABEL: @absl(
// CHECK:   [[ABS:%.*]] = call i64 @llvm.abs.i64(i64 [[X:%.*]], i1 true)
//
  return __builtin_labs(x);
}

long long absll(long long x) {
// CHECK-LABEL: @absll(
// CHECK:   [[ABS:%.*]] = call i64 @llvm.abs.i64(i64 [[X:%.*]], i1 true)
//
  return __builtin_llabs(x);
}

