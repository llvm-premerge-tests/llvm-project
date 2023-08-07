// RUN: %clang_cc1 %s -fsyntax-only -verify

/// Zero-sized structs should not crash.
int b() {
  struct {      } a[10];
  __builtin_memcpy(&a[2], a, 2); // expected-warning {{buffer has size 0, but size argument is 2}}
  return 0;
}
