// RUN: %clang_cc1 -emit-llvm -o - -std=c23 %s | FileCheck %s
// expected-no-diagnostics

#include <stdckdint.h>
#include <stdint.h>
// CHECK-LABEL: define dso_local void @test_add_overflow_to64() #0 {
// CHECK:  entry:
// CHECK:    %result64 = alloca i64, align 8
// CHECK:    %flag_add = alloca i8, align 1
// CHECK:    store i64 0, ptr %result64, align 8
// CHECK:    %0 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 2147483647, i64 1)
// CHECK:    %1 = extractvalue { i64, i1 } %0, 1 
// CHECK:    %2 = extractvalue { i64, i1 } %0, 0
// CHECK:    store i64 %2, ptr %result64, align 8
// CHECK:    %frombool = zext i1 %1 to i8
// CHECK:    store i8 %frombool, ptr %flag_add, align 1
// CHECK:    ret void
// CHECK:  }
void test_add_overflow_to64() {
    int64_t result64 = 0;
    bool flag_add = ckd_add(&result64, INT32_MAX, 1);
}

// CHECK-LABEL: define dso_local void @test_sub_overflow() #0 {
// CHECK:  entry:
// CHECK:    %result32 = alloca i32, align 4
// CHECK:    %flag_sub = alloca i8, align 1
// CHECK:    store i32 0, ptr %result32, align 4
// CHECK:    %0 = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 2147483647, i32 -1)
// CHECK:    %1 = extractvalue { i32, i1 } %0, 1 
// CHECK:    %2 = extractvalue { i32, i1 } %0, 0
// CHECK:    store i32 %2, ptr %result32, align 4
// CHECK:    %frombool = zext i1 %1 to i8 
// CHECK:    store i8 %frombool, ptr %flag_sub, align 1 
// CHECK:    ret void
// CHECK:  }
void test_sub_overflow() {
    int32_t result32 = 0;
    bool flag_sub = ckd_sub(&result32, INT32_MAX, -1);
}

// CHECK-LABEL: define dso_local void @test_mul_normal() #0 {
// CHECK:  entry:
// CHECK:    %a = alloca i32, align 4
// CHECK:    %result = alloca i32, align 4
// CHECK:    %flag_mul = alloca i8, align 1
// CHECK:    store i32 3, ptr %a, align 4
// CHECK:    store i32 0, ptr %result, align 4
// CHECK:    %0 = load i32, ptr %a, align 4
// CHECK:    %1 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %0, i32 2)
// CHECK:    %2 = extractvalue { i32, i1 } %1, 1 
// CHECK:    %3 = extractvalue { i32, i1 } %1, 0
// CHECK:    store i32 %3, ptr %result, align 4
// CHECK:    %frombool = zext i1 %2 to i8 
// CHECK:    store i8 %frombool, ptr %flag_mul, align 1 
// CHECK:    ret void
// CHECK:  }
void test_mul_normal() {
    int a = 3;
    int result = 0;
    bool flag_mul = ckd_mul(&result, a, 2);
}
