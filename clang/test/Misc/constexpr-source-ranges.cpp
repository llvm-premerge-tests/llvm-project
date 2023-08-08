// RUN: not %clang_cc1 -fsyntax-only -fdiagnostics-print-source-range-info -fcxx-exceptions %s 2>&1 | FileCheck %s
// RUN: not %clang_cc1 -fsyntax-only -fexperimental-new-constant-interpreter -fdiagnostics-print-source-range-info -fcxx-exceptions %s 2>&1 | FileCheck %s

constexpr int f() {
  throw 1;
  return 0;
}

// CHECK: constexpr-source-ranges.cpp:5:3:{5:3-5:10}


constexpr int I = 12;
constexpr const int *P = &I;
constexpr long L = (long)P;
// CHECK: constexpr-source-ranges.cpp:14:20:{14:20-14:27}

namespace overflow {
// CHECK:      :{[[@LINE+1]]:9-[[@LINE+1]]:29}: warning: overflow
int x = -1 + __INT_MAX__ + 2 + 3;
// CHECK:      :{[[@LINE+1]]:9-[[@LINE+1]]:19}: warning: overflow
int a = -(1 << 31) + 1;
}
