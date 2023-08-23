// RUN: %clang_cc1 -fsyntax-only -verify %s

struct Foo {
  int a;
};

struct Bar {
  int a;
  Bar();
};

[[clang::unnamed_addr]] const int i = 8;

[[clang::unnamed_addr]] int i2 = 8; // expected-warning{{unnamed_addr should only be used on const POD (plain old data) globals}}

[[clang::unnamed_addr]] const Foo j = {2};

[[clang::unnamed_addr]] Foo j2 = {2}; // expected-warning{{unnamed_addr should only be used on const POD (plain old data) globals}}

[[clang::unnamed_addr]] const Bar k; // expected-warning{{unnamed_addr should only be used on const POD (plain old data) globals}}
