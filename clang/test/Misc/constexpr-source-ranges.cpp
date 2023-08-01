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


constexpr int div(bool a, bool b) {
  return 1 / (int)b;
}
constexpr int ints(int a, int b, int c, int d) {
  return 1;
}
static_assert(ints(1, div(true, false), 2, div(false, true)) == 1, "");
// CHECK: constexpr-source-ranges.cpp:24:23:{24:23-24:39}
