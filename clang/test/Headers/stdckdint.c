// RUN: %clang_cc1 -emit-llvm -verify -fgnuc-version=4.2.1 -std=c23 %s -o - | FileCheck %s
// expected-no-diagnostics
#include <stdckdint.h>

// CHECK-LABEL: define dso_local zeroext i1 @test_ckd_add() #0 {
// CHECK:  entry:
// CHECK:    %result = alloca i32, align 4
// CHECK:    %0 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 -1073741826, i32 -1073741826)
// CHECK:    %1 = extractvalue { i32, i1 } %0, 1
// CHECK:    %2 = extractvalue { i32, i1 } %0, 0
// CHECK:    store i32 %2, ptr %result, align 4
// CHECK:    ret i1 %1
// CHECK:    }
// CHECK:    ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none)
// CHECK:    declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) #1
bool test_ckd_add() {
  int result;
  return ckd_add(&result, -1073741826, -1073741826);
}

// CHECK-LABEL: define dso_local zeroext i1 @test_ckd_sub() #0 {
// CHECK:  entry:
// CHECK:    %result = alloca i32, align 4
// CHECK:    %0 = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 -1073741826, i32 1073741826)
// CHECK:    %1 = extractvalue { i32, i1 } %0, 1
// CHECK:    %2 = extractvalue { i32, i1 } %0, 0
// CHECK:    store i32 %2, ptr %result, align 4
// CHECK:    ret i1 %1
// CHECK:    }
// CHECK:    declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) #1
bool test_ckd_sub() {
  int result;
  return ckd_sub(&result, -1073741826, 1073741826);
}

// CHECK-LABEL: define dso_local zeroext i1 @test_ckd_mul() #0 {
// CHECK:  entry:
// CHECK:    %result = alloca i32, align 4
// CHECK:    %0 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 -1073741826, i32 2)
// CHECK:    %1 = extractvalue { i32, i1 } %0, 1
// CHECK:    %2 = extractvalue { i32, i1 } %0, 0
// CHECK:    store i32 %2, ptr %result, align 4
// CHECK:    ret i1 %1
// CHECK:    }
// CHECK:    declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32) #1
bool test_ckd_mul() {
  int result;
  return ckd_mul(&result, -1073741826, 2);
}
