// RUN: %clang_cc1 %s -std=c++11 -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions | FileCheck %s

void external();

// CHECK-LABEL: @_Z6targetv(
// CHECK: call unwindabort void @_Z8externalv()
// CHECK: ret void
void target() noexcept
{
  external();
}

// CHECK-LABEL: @_Z7reversev(
// CHECK: call void @_Z8externalv()
// CHECK: ret void
void reverse() noexcept(false)
{
  external();
}
