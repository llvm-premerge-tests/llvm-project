// RUN: %check_clang_tidy -std=c++11,c++14,c++17,c++20 %s bugprone-exception-escape %t -- \
// RUN:     -- -fexceptions

void rethrower() {
    throw;
}

void callsRethrower() {
    rethrower();
}

void callsRethrowerNoexcept() noexcept {
    rethrower();
}

int throwsAndCallsRethrower() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'throwsAndCallsRethrower' which should not throw exceptions
// CHECK-MESSAGES: :[[@LINE+2]]:9: note: may throw 'int' here
    try {
        throw 1;
    } catch(...) {
        rethrower();
    }
}

int throwsAndCallsCallsRethrower() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:5: warning: an exception may be thrown in function 'throwsAndCallsCallsRethrower' which should not throw exceptions
// CHECK-MESSAGES: :[[@LINE+2]]:9: note: may throw 'int' here
    try {
        throw 1;
    } catch(...) {
        callsRethrower();
    }
}

void rethrowerNoexcept() noexcept {
    throw;
}

void throwInt() {
  throw 5;
}

void rethrowInt() noexcept {
// CHECK-MESSAGES: :[[@LINE-1]]:6: warning: an exception may be thrown in function 'rethrowInt' which should not throw exceptions [bugprone-exception-escape]
// CHECK-MESSAGES: :[[@LINE-5]]:3: note: may throw 'int' here
  try {
    throwInt();
  } catch(int) {
    throw;
  } catch(...) {
  }
}
