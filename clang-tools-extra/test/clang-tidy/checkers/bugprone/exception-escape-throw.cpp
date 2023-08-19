// RUN: %check_clang_tidy -std=c++11,c++14 %s bugprone-exception-escape %t -- -- -fexceptions

void throwing_throw_nothing() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'throwing_throw_nothing' which should not throw exceptions
// CHECK-MESSAGES: :[[@LINE+1]]:3: note: may throw 'int' here
  throw 1;
}

void explicit_int_thrower() throw(int);

void implicit_int_thrower() {
  throw 5;
}

void indirect_implicit() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_implicit' which should not throw exceptions
// CHECK-MESSAGES: :[[@LINE-5]]:3: note: may throw 'int' here
  implicit_int_thrower();
}

void indirect_explicit() throw() {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'indirect_explicit' which should not throw exceptions
// CHECK-MESSAGES: :[[@LINE-14]]:29: note: may throw 'int' here
// CHECK-MESSAGES: :[[@LINE-3]]:6: note: may throw unknown exceptions here
  explicit_int_thrower();
}

struct super_throws {
  super_throws() throw(int) { throw 42; }
};

struct sub_throws : super_throws {
  sub_throws() throw() : super_throws() {}
  // CHECK-MESSAGES: :[[@LINE-1]]:3: warning: an exception may be thrown in function 'sub_throws' which should not throw exceptions
  // CHECK-MESSAGES: :[[@LINE-6]]:31: note: may throw 'int' here
};
