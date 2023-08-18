// RUN: %clang_cc1 -emit-llvm -Wno-gcc-compat -O1 -disable-llvm-passes -o - %s | FileCheck %s

int v01 __attribute__((linkage("external")));
int v02 __attribute__((linkage("available_externally")));
int v03 __attribute__((linkage("linkonce")));
int v04 __attribute__((linkage("linkonce_odr")));
int v05 __attribute__((linkage("weak")));
int v06 __attribute__((linkage("weak_odr")));
int v07[3] __attribute__((linkage("appending")));
int v08 __attribute__((linkage("internal")));
int v09 __attribute__((linkage("private")));
int v10 __attribute__((linkage("external_weak")));
int v11 __attribute__((linkage("common")));

// CHECK: @v01 = global i32 0
// CHECK: @v02 = available_externally global i32 0
// CHECK: @v03 = linkonce global i32 0
// CHECK: @v04 = linkonce_odr global i32 0
// CHECK: @v05 = weak global i32 0
// CHECK: @v06 = weak_odr global i32 0
// CHECK: @v07 = appending global [3 x i32] zeroinitializer
// CHECK: @v08 = internal global i32 0
// CHECK: @v09 = private global i32 0
// CHECK: @v10 = extern_weak global i32 0
// CHECK: @v11 = common global i32 0

void f01(void) __attribute__((linkage("external"))) {};
void f02(void) __attribute__((linkage("available_externally"))) {};
void f03(void) __attribute__((linkage("linkonce"))) {};
void f04(void) __attribute__((linkage("linkonce_odr"))) {};
void f05(void) __attribute__((linkage("weak"))) {};
void f06(void) __attribute__((linkage("weak_odr"))) {};
// appending is only applicable to variables
void f08(void) __attribute__((linkage("internal"))) {};
void f09(void) __attribute__((linkage("private"))) {};
void f10(void) __attribute__((linkage("external_weak"))) {};
// common is only applicable to variables

// CHECK: define dso_local void @f01()
// CHECK: define available_externally void @f02()
// CHECK: define linkonce void @f03()
// CHECK: define linkonce_odr void @f04()
// CHECK: define weak void @f05()
// CHECK: define weak_odr void @f06()
// CHECK: define internal void @f08()
// CHECK: define private void @f09()
// CHECK: define extern_weak void @f10()
