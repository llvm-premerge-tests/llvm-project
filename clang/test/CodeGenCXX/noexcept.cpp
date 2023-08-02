// RUN: %clang_cc1 %s -triple=x86_64-apple-darwin10 -emit-llvm -o - -fcxx-exceptions -fexceptions -std=c++11 | FileCheck %s

// rdar://11904428
//   Ensure that we call __cxa_begin_catch before calling
//   std::terminate in a noexcept function.
namespace test0 {
  void foo();

  struct A {
    A();
    ~A();
  };

  void test() noexcept {
    A a;
    foo();
  }
}

// CHECK-LABEL:    define{{.*}} void @_ZN5test04testEv()
// CHECK:      call unwindabort void @_ZN5test01AC1Ev(
// CHECK:      call unwindabort void @_ZN5test03fooEv()
//   Destructors don't throw by default in C++11.
// CHECK:      call void @_ZN5test01AD1Ev(
