// RUN: %clang_cc1 -triple x86_64-linux -fstrict-flex-arrays=3 -fsanitize=array-bounds \
// RUN:     -fsyntax-only -verify %s

#define __counted_by(f)  __attribute__((counted_by(f)))

struct bar;

struct foo {
  int bork;
  struct bar *fam[] __counted_by(not_found); // expected-warning {{counted_by field 'not_found' not found}}
};

struct baz {
  int bork;
  struct bar *self[] __counted_by(self); // expected-error {{counted_by field 'self' cannot refer to the flexible array}}
};

struct qux {
  double non_integer; // expected-error {{counted_by field 'non_integer' is not an integral type}}
  struct bar *fam[] __counted_by(non_integer); // expected-note {{counted_by field 'non_integer' declared here}}
};
