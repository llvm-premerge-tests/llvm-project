// REQUIRES: arch=x86_64
//
// RUN: %clangxx -fsanitize=builtin -fno-sanitize-recover=signed-integer-overflow -w %s -O3 -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=RECOVER
// RUN: %clangxx -fsanitize=signed-integer-overflow -fno-sanitize-recover=builtin -w %s -O3 -o %t
// RUN: %run %t 2>&1 | FileCheck %s --check-prefix=RECOVER
//
// RUN: %clangxx -fsanitize=builtin -fsanitize=signed-integer-overflow -fno-sanitize-recover=builtin -w %s -O3 -o %t.abort
// RUN: not %run %t.abort 2>&1 | FileCheck %s --check-prefix=ABORT
// RUN: %clangxx -fsanitize=builtin -fsanitize=signed-integer-overflow -fno-sanitize-recover=signed-integer-overflow -w %s -O3 -o %t.abort
// RUN: not %run %t.abort 2>&1 | FileCheck %s --check-prefix=ABORT

#include <limits.h>

int main() {
  // ABORT: abs.cpp:[[@LINE+2]]:17: runtime error: passing a minimal signed integer to abs(), which results in overflow
  // RECOVER: abs.cpp:[[@LINE+1]]:17: runtime error: passing a minimal signed integer to abs(), which results in overflow
  __builtin_abs(INT_MIN);

  // RECOVER: abs.cpp:[[@LINE+1]]:18: runtime error: passing a minimal signed integer to abs(), which results in overflow
  __builtin_labs(LONG_MIN);

  // RECOVER: abs.cpp:[[@LINE+1]]:19: runtime error: passing a minimal signed integer to abs(), which results in overflow
  __builtin_llabs(LLONG_MIN);
  return 0;
}
