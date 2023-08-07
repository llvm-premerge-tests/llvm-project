// RUN: %clang_cc1 -emit-llvm  -fgnuc-version=4.2.1 -std=gnu++11 %s -o - | FileCheck --check-prefix=CHECK-NEXT %s


#include <stdckdint.h>
// CHECK-LABEL: define dso_local zeroext i1 @test_ckd_add() #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    %0 = call { i32, i1 } @llvm.sadd.with.overflow.i32(i32 -1073741826, i32 -1073741826)
// CHECK-NEXT:    %1 = extractvalue { i32, i1 } %0, 1 
// CHECK-NEXT:    store i32 %2, ptr %result, align 4 
// CHECK-NEXT:    ret i1 %1 
// CHECK-NEXT:    } 
// CHECK-NEXT:    ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none) 
// CHECK-NEXT:    declare { i32, i1 } @llvm.sadd.with.overflow.i32(i32, i32) #1 
bool test_ckd_add() {
  int result;
  return ckd_add(&result, -1073741826, -1073741826);
}

// CHECK-LABEL: define dso_local zeroext i1 @test_ckd_sub() #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    %0 = call { i32, i1 } @llvm.ssub.with.overflow.i32(i32 -1073741826, i32 1073741826)
// CHECK-NEXT:    %1 = extractvalue { i32, i1 } %0, 1 
// CHECK-NEXT:    store i32 %2, ptr %result, align 4 
// CHECK-NEXT:    ret i1 %1 
// CHECK-NEXT:    } 
// CHECK-NEXT:    ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none) 
// CHECK-NEXT:    declare { i32, i1 } @llvm.ssub.with.overflow.i32(i32, i32) #1 
bool test_ckd_sub() {
  int result;
  return ckd_sub(&result, -1073741826, 1073741826);
}

// CHECK-LABEL: define dso_local zeroext i1 @test_ckd_mul() #0 {
// CHECK-NEXT:  entry:
// CHECK-NEXT:    %0 = call { i32, i1 } @llvm.smul.with.overflow.i32(i32 -1073741826, i32 2)
// CHECK-NEXT:    %1 = extractvalue { i32, i1 } %0, 1 
// CHECK-NEXT:    store i32 %2, ptr %result, align 4
// CHECK-NEXT:    ret i1 %1 
// CHECK-NEXT:    } 
// CHECK-NEXT:    ; Function Attrs: nocallback nofree nosync nounwind speculatable willreturn memory(none) 
// CHECK-NEXT:    declare { i32, i1 } @llvm.smul.with.overflow.i32(i32, i32) #1 
bool test_ckd_mul() {
  int result;
  return ckd_mul(&result, -1073741826, 2);
}
