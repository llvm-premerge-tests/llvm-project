// RUN: %clang_cc1 -fexperimental-new-constant-interpreter -verify %s
// RUN: %clang_cc1 -verify=ref %s

/// expected-no-diagnostics

_Static_assert(1, "");
_Static_assert(0 != 1, "");
_Static_assert(1.0 == 1.0, "");
_Static_assert( (5 > 4) + (3 > 2) == 2, "");

/// FIXME: Should also be rejected in the new interpreter
int a = (1 == 1 ? 5 : 3);
_Static_assert(a == 5, ""); // ref-error {{ not an integral constant expression}}

const int b = 3;
_Static_assert(b == 3, "");
