// RUN: %clang_cc1 -std=c++17 %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s --implicit-check-not='call{{.*}}dtor'
// RUN: %clang_cc1 -std=c++23 %s -triple x86_64-linux-gnu -emit-llvm -o - | FileCheck %s --check-prefixes=CHECK-CXX23,CHECK-CXX23-NEXT

namespace std {
  typedef decltype(sizeof(int)) size_t;

  template <class E>
  struct initializer_list {
    const E *begin;
    size_t   size;
    initializer_list() : begin(nullptr), size(0) {}
  };
}

void then();

struct dtor {
  ~dtor();
};

dtor ctor();

auto &&lambda = [a = {ctor()}] {};
// CHECK-LABEL: define
// CHECK: call {{.*}}ctor
// CHECK: call {{.*}}atexit{{.*}}global_array_dtor

// CHECK-LABEL: define{{.*}}global_array_dtor
// CHECK: call {{.*}}dtor

// [lifetime extension occurs if the object was obtained by]
//  -- a temporary materialization conversion
// CHECK-LABEL: ref_binding
void ref_binding() {
  // CHECK: call {{.*}}ctor
  auto &&x = ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- ( expression )
// CHECK-LABEL: parens
void parens() {
  // CHECK: call {{.*}}ctor
  auto &&x = ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- subscripting of an array
// CHECK-LABEL: array_subscript_1
void array_subscript_1() {
  using T = dtor[1];
  // CHECK: call {{.*}}ctor
  auto &&x = T{ctor()}[0];
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: array_subscript_2
void array_subscript_2() {
  using T = dtor[1];
  // CHECK: call {{.*}}ctor
  auto &&x = ((dtor*)T{ctor()})[0];
  // CHECK: call {{.*}}dtor
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}

struct with_member { dtor d; ~with_member(); };
struct with_ref_member { dtor &&d; ~with_ref_member(); };

//  -- a class member access using the . operator [...]
// CHECK-LABEL: member_access_1
void member_access_1() {
  // CHECK: call {{.*}}ctor
  auto &&x = with_member{ctor()}.d;
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}with_member
  // CHECK: }
}
// CHECK-LABEL: member_access_2
void member_access_2() {
  // CHECK: call {{.*}}ctor
  auto &&x = with_ref_member{ctor()}.d;
  // CHECK: call {{.*}}with_ref_member
  // CHECK: call {{.*}}dtor
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}
// CHECK-LABEL: member_access_3
void member_access_3() {
  // CHECK: call {{.*}}ctor
  auto &&x = (&(const with_member&)with_member{ctor()})->d;
  // CHECK: call {{.*}}with_member
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}

//  -- a pointer-to-member operation using the .* operator [...]
// CHECK-LABEL: member_ptr_access_1
void member_ptr_access_1() {
  // CHECK: call {{.*}}ctor
  auto &&x = with_member{ctor()}.*&with_member::d;
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}with_member
  // CHECK: }
}
// CHECK-LABEL: member_ptr_access_2
void member_ptr_access_2() {
  // CHECK: call {{.*}}ctor
  auto &&x = (&(const with_member&)with_member{ctor()})->*&with_member::d;
  // CHECK: call {{.*}}with_member
  // CHECK: call {{.*}}then
  then();
  // CHECK: }
}

//  -- a [named] cast [...]
// CHECK-LABEL: static_cast
void test_static_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = static_cast<dtor&&>(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: const_cast
void test_const_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = const_cast<dtor&&>(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: reinterpret_cast
void test_reinterpret_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = reinterpret_cast<dtor&&>(static_cast<dtor&&>(ctor()));
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: dynamic_cast
void test_dynamic_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = dynamic_cast<dtor&&>(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- [explicit cast notation is defined in terms of the above]
// CHECK-LABEL: c_style_cast
void c_style_cast() {
  // CHECK: call {{.*}}ctor
  auto &&x = (dtor&&)ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: function_style_cast
void function_style_cast() {
  // CHECK: call {{.*}}ctor
  using R = dtor&&;
  auto &&x = R(ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- a conditional operator
// CHECK-LABEL: conditional
void conditional(bool b) {
  // CHECK: call {{.*}}ctor
  // CHECK: call {{.*}}ctor
  auto &&x = b ? (dtor&&)ctor() : (dtor&&)ctor();
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

//  -- a comma expression
// CHECK-LABEL: comma
void comma() {
  // CHECK: call {{.*}}ctor
  auto &&x = (true, (dtor&&)ctor());
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}


// This applies recursively: if an object is lifetime-extended and contains a
// reference, the referent is also extended.
// CHECK-LABEL: init_capture_ref
void init_capture_ref() {
  // CHECK: call {{.*}}ctor
  auto x = [&a = (const dtor&)ctor()] {};
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: init_capture_ref_indirect
void init_capture_ref_indirect() {
  // CHECK: call {{.*}}ctor
  auto x = [&a = (const dtor&)ctor()] {};
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}
// CHECK-LABEL: init_capture_init_list
void init_capture_init_list() {
  // CHECK: call {{.*}}ctor
  auto x = [a = {ctor()}] {};
  // CHECK: call {{.*}}then
  then();
  // CHECK: call {{.*}}dtor
  // CHECK: }
}

namespace P2718R0 {

// Test basic
struct A {
  int a[3] = {1, 2, 3};
  A() {}
  ~A() {}
  const int *begin() const { return a; }
  const int *end() const { return a + 3; }
  A& r() { return *this; }
  A g() { return A(); }
};

A g() { return A(); }
const A &f1(const A &t) { return t; }

void test1() {
  [[maybe_unused]] int sum = 0;
  // CHECK-CXX23: void @_ZN7P2718R05test1Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : f1(g()))
    sum += e;
}

struct B : A {};
int (&f(const A *))[3];
const A *g(const A &);
void bar(int) {}

void test2() {
  // CHECK-CXX23: void @_ZN7P2718R05test2Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01BD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : f(g(B())))
    bar(e);
}

// Test discard statement.
struct LockGuard {
    LockGuard() {}
    ~LockGuard() {}
};

void test3() {
  int v[] = {42, 17, 13};

  // CHECK-CXX23: void @_ZN7P2718R05test3Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R09LockGuardD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for ([[maybe_unused]] int x : static_cast<void>(LockGuard()), v)
    LockGuard guard;
  
  // CHECK-CXX23: for.cond.cleanup11:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R09LockGuardD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end17
  for ([[maybe_unused]] int x : (void)LockGuard(), v)
    LockGuard guard;
  
  // CHECK-CXX23: for.cond.cleanup27:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R09LockGuardD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end33
  for ([[maybe_unused]] int x : LockGuard(), v)
    LockGuard guard;
}

// Test default arg
int (&default_arg_fn(const A & = A()))[3];
void test4() {

  // CHECK-CXX23: void @_ZN7P2718R05test4Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : default_arg_fn()) 
    bar(e);
}

struct DefaultA {
  DefaultA() {}
  ~DefaultA() {}
};

A foo(const A&, const DefaultA &Default = DefaultA()) {
  return A();
}

void test5() {
  // CHECK-CXX23: void @_ZN7P2718R05test5Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R08DefaultAD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R08DefaultAD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R08DefaultAD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : default_arg_fn(foo(foo(foo(A())))))
    bar(e);
}

struct C : public A {
  C() {}
  C(int, const C &, const DefaultA & = DefaultA()) {}
};

void test6() {
  // CHECK-CXX23: void @_ZN7P2718R05test6Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01CD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R08DefaultAD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01CD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R08DefaultAD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01CD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R08DefaultAD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01CD1Ev(
  // CHECK-CXX23: br label %for.end
  for (auto e : C(0, C(0, C(0, C()))))
    bar(e);
}

// Test member call
void test7() {
  // CHECK-CXX23: void @_ZN7P2718R05test7Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : g().r().g().r().g().r().g())
    bar(e);
}

// Test basic && dependent context
template <typename T> T dg() { return T(); }
template <typename T> const T &df1(const T &t) { return t; }

void test8() {
  [[maybe_unused]] int sum = 0;
  // CHECK-CXX23: void @_ZN7P2718R05test8Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : df1(dg<A>()))
    sum += e;
}

template <typename T> int (&df2(const T *))[3];
const A *dg2(const A &);

void test9() {
  // CHECK-CXX23: void @_ZN7P2718R05test9Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01BD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : df2(dg2(B())))
    bar(e);
}

// Test discard statement && dependent context
void test10() {
  int v[] = {42, 17, 13};

  // CHECK-CXX23: void @_ZN7P2718R06test10Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R09LockGuardD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for ([[maybe_unused]] int x : static_cast<void>(df1(LockGuard())), v)
    LockGuard guard;
  
  // CHECK-CXX23: for.cond.cleanup12:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R09LockGuardD1Ev(
  // CHECK-CXX23-NEXT: br label %for.inc16
  for ([[maybe_unused]] int x : (void)df1(LockGuard()), v)
    LockGuard guard;
  
  // CHECK-CXX23: for.cond.cleanup31:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R09LockGuardD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R09LockGuardD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end37
  for ([[maybe_unused]] int x : df1(LockGuard()), df1(LockGuard()), v)
    LockGuard guard;
}

// Test default argument && dependent context
template <typename T> int (&default_arg_fn2(const T & = T()))[3];
void test11() {
  // CHECK-CXX23: void @_ZN7P2718R06test11Ev()
  // CHECK-CXX23-NEXT: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : default_arg_fn2<A>()) 
    bar(e);
}

template <typename T> A foo2(const T&, const DefaultA &Default = DefaultA());

void test12() {
  // CHECK-CXX23: void @_ZN7P2718R06test12Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R08DefaultAD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R08DefaultAD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R08DefaultAD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : default_arg_fn2(foo2(foo2(foo2(A())))))
    bar(e);
}

// Test member call && dependent context
void test13() {

  // CHECK-CXX23: void @_ZN7P2718R06test13Ev()
  // CHECK-CXX23: for.cond.cleanup:
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: call void @_ZN7P2718R01AD1Ev(
  // CHECK-CXX23-NEXT: br label %for.end
  for (auto e : dg<A>().r().g().r().g().r().g())
    bar(e);
}
} // namespace P2718R0

