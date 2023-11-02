// RUN: %clang_cc1 -triple %itanium_abi_triple -fsyntax-only %s -fcxx-exceptions -fassume-nothrow-exception-dtor -verify

namespace test1 {
struct A {};
struct B { ~B() noexcept(false); };
struct B1 : B {};
struct B2 { B b; };
struct C { virtual void f(); } c;
void run() {
  throw A();
  throw B();  // expected-error{{thrown object has a potentially-throwing destructor}}
  throw B1(); // expected-error{{thrown object has a potentially-throwing destructor}}
  throw B2(); // expected-error{{thrown object has a potentially-throwing destructor}}
  throw c;
}
}
