// RUN: %clang_cc1 -emit-llvm -o - -std=c23 %s | FileCheck %s

#include <stdckdint.h>
#include <stdint.h>
// CHECK-LABEL: define dso_local void @test_add_overflow_to64() #0 {
// CHECK:  entry:
// CHECK:    %[[RES64:.*]] = alloca i64, align 8
// CHECK:    %[[FLAG_ADD:.*]] = alloca i8, align 1
// CHECK:    store i64 0, ptr %result64, align 8
// CHECK:    %0 = call { i64, i1 } @llvm.sadd.with.overflow.i64(i64 2147483647, i64 1)
// CHECK:    %1 = extractvalue { i64, i1 } %0, 1 
// CHECK:    %2 = extractvalue { i64, i1 } %0, 0
// CHECK:    store i64 %2, ptr %[[RES64]], align 8
// CHECK:    %frombool = zext i1 %1 to i8
// CHECK:    store i8 %frombool, ptr %[[FLAG_ADD:.*]], align 1
// CHECK:    ret void
// CHECK:  }
void test_add_overflow_to64() {
  int64_t result64 = 0;
  bool flag_add = ckd_add(&result64, INT32_MAX, 1);
}

// CHECK-LABEL: define dso_local void @test_sub_overflow() #0 {
// CHECK:  entry:
// CHECK:    %[[RES32:.*]] = alloca i32, align 4
// CHECK:    %[[FLAG_SUB:.*]] = alloca i8, align 1
// CHECK:    store i32 0, ptr %[[RES32:.*]], align 4
// CHECK:    %0 = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 2147483647, i32 -1)
// CHECK:    %1 = extractvalue { i32, i1 } %0, 1 
// CHECK:    %2 = extractvalue { i32, i1 } %0, 0
// CHECK:    store i32 %2, ptr %[[RES32:.*]], align 4
// CHECK:    %frombool = zext i1 %1 to i8 
// CHECK:    store i8 %frombool, ptr %[[FLAG_SUB:.*]], align 1
// CHECK:    ret void
// CHECK:  }
void test_sub_overflow() {
  int32_t result32 = 0;
  bool flag_sub = ckd_sub(&result32, INT32_MAX, -1);
}

// CHECK-LABEL: define dso_local void @test_mul_normal() #0 {
// CHECK:  entry:
// CHECK:    %[[A:.*]] = alloca i32, align 4
// CHECK:    %[[RES:.*]] = alloca i32, align 4
// CHECK:    %[[FLAG_MUL:.*]] = alloca i8, align 1
// CHECK:    store i32 3, ptr %a, align 4
// CHECK:    store i32 0, ptr %[[RES:.*]], align 4
// CHECK:    %0 = load i32, ptr %[[A:.*]], align 4
// CHECK:    %1 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 %0, i32 2)
// CHECK:    %2 = extractvalue { i32, i1 } %1, 1 
// CHECK:    %3 = extractvalue { i32, i1 } %1, 0
// CHECK:    store i32 %3, ptr %[[RES:.*]], align 4
// CHECK:    %frombool = zext i1 %2 to i8 
// CHECK:    store i8 %frombool, ptr %[[FLAG_MUL:.*]], align 1
// CHECK:    ret void
// CHECK:  }
void test_mul_normal() {
  int a = 3;
  int result = 0;
  bool flag_mul = ckd_mul(&result, a, 2);
}
