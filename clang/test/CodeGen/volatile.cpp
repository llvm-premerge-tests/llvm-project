// RUN: %clang_cc1 -O2 -triple=x86_64-unknown-linux-gnu -emit-llvm %s -o -  | FileCheck %s -check-prefix CHECK

int x ;
void f0() {
    const_cast<volatile int&>(x) = const_cast<volatile int&>(x) + 1;
//  CHECK: %0 = load volatile i32, ptr @x
//  CHECK: %add = add nsw i32 %0, 1
//  CHECK: store volatile i32 %add, ptr @x
    static_cast<volatile int&>(x) = static_cast<volatile int&>(x) + 1;
//  CHECK: %1 = load volatile i32, ptr @x
//  CHECK: %add1 = add nsw i32 %1, 1
//  CHECK: store volatile i32 %add1, ptr @x
}
