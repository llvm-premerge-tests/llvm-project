// RUN: %clang_cc1 -triple x86_64-unknown-unknown -fsyntax-only -verify %s

struct Bar {int a;};
const Bar arr[2] = {{1}};

struct Foo {};

const int b = 2;

void foo(int a) {
  Foo *foo_array;
  foo_array = new Foo[arr[0].a];
}

void Test(int N) {       // expected-note {{declared here}}
  int arr[N];            // expected-warning {{variable length arrays are a Clang extension}} \\
                            expected-note {{function parameter 'N' with unknown value cannot be used in a constant expression}}
  decltype([&arr]{}) *p; // expected-error {{lambda expression in an unevaluated operand}}
}
