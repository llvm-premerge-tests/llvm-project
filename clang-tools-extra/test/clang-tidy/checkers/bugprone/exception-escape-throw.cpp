// RUN: %check_clang_tidy -std=c++11,c++14 %s bugprone-exception-escape %t -- -- -fexceptions

void throwing_throw_nothing() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throwing_throw_nothing' which should not throw exceptions
  throw 1;
}

void explicit_int_thrower() throw(int);

void implicit_int_thrower() {
  throw 5;
}

void indirect_implicit() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_implicit' which should not throw exceptions
  implicit_int_thrower();
}

void indirect_explicit() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_explicit' which should not throw exceptions
  explicit_int_thrower();
}

struct super_throws {
  super_throws() throw(int) { throw 42; }
};

struct sub_throws : super_throws {
  sub_throws() throw() : super_throws() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'sub_throws' which should not throw exceptions
};

namespace PR55143 { namespace PR40583 {

struct test_explicit_throw {
    test_explicit_throw() throw(int) { throw 42; }
    test_explicit_throw(const test_explicit_throw&) throw(int) { throw 42; }
    test_explicit_throw(test_explicit_throw&&) throw(int) { throw 42; }
    test_explicit_throw& operator=(const test_explicit_throw&) throw(int) { throw 42; }
    test_explicit_throw& operator=(test_explicit_throw&&) throw(int) { throw 42; }
    ~test_explicit_throw() throw(int) { throw 42; }
};

struct test_implicit_throw {
    test_implicit_throw() { throw 42; }
    test_implicit_throw(const test_implicit_throw&) { throw 42; }
    test_implicit_throw(test_implicit_throw&&) { throw 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'test_implicit_throw' which should not throw exceptions
    test_implicit_throw& operator=(const test_implicit_throw&) { throw 42; }
    test_implicit_throw& operator=(test_implicit_throw&&) { throw 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:26: warning: an exception may be thrown in function 'operator=' which should not throw exceptions
    ~test_implicit_throw() { throw 42; }
    // CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function '~test_implicit_throw' which should not throw exceptions
};

}}
