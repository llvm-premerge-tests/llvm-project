// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 -triple riscv64 -target-feature +v \
// RUN:   -emit-llvm %s -o - | FileCheck -check-prefix=CHECK-LLVM %s

#include <riscv_vector.h>

// CHECK-LLVM: void @call1(<vscale x 4 x i32> %v0, <vscale x 8 x i32> %v1.coerce0, <vscale x 8 x i32> %v1.coerce1, <vscale x 8 x i32> %v2, <vscale x 2 x i32> %v3)
void call1(vint32m2_t v0, vint32m4x2_t v1, vint32m4_t v2, vint32m1_t v3) {}

// CHECK-LLVM: void @call2(<vscale x 2 x i32> %v0.coerce0, <vscale x 2 x i32> %v0.coerce1, <vscale x 2 x i32> %v0.coerce2, <vscale x 8 x i32> %v1.coerce0, <vscale x 8 x i32> %v1.coerce1, <vscale x 8 x i32> %v2, ptr noundef %0)
void call2(vint32m1x3_t v0, vint32m4x2_t v1, vint32m4_t v2, vint32m2_t v3) {}

// CHECK-LLVM: void @call3(<vscale x 8 x i32> %v0.coerce0, <vscale x 8 x i32> %v0.coerce1, ptr noundef %0, <vscale x 8 x i32> %v2.coerce0, <vscale x 8 x i32> %v2.coerce1)
void call3(vint32m4x2_t v0, vint32m1_t v1, vint32m4x2_t v2) {}

// CHECK-LLVM: void @call4(<vscale x 16 x i32> %v0, ptr noundef %0, <vscale x 16 x i32> %v2)
void call4(vint32m8_t v0, vint32m1_t v1, vint32m8_t v2) {}

// CHECK-LLVM: void @call5(ptr noundef %0, <vscale x 16 x i32> %v1, ptr noundef %1, <vscale x 16 x i32> %v3)
void call5(vint32m1_t v0, vint32m8_t v1, vint32m1_t v2, vint32m8_t v3) {}

// CHECK-LLVM: void @call6(<vscale x 1 x i8> %v0, <vscale x 64 x i8> %v1, <vscale x 2 x i32> %v2, <vscale x 1 x i8> %v3)
void call6(vint8mf8_t v0, vint8m8_t v1, vint32m1_t v2, vint8mf8_t v3) {}

// CHECK-LLVM: void @call7(ptr noundef %0, <vscale x 64 x i8> %v1, <vscale x 16 x i32> %v2, ptr noundef %1)
void call7(vint8mf8_t v0, vint8m8_t v1, vint32m8_t v2, vint8mf8_t v3) {}
