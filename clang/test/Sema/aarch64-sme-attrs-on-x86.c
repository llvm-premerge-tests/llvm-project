// RUN: %clang_cc1 -triple x86_64-none-linux-gnu -target-feature +sme -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

extern int normal_callee();

// expected-warning@+1 {{unknown attribute 'arm_streaming' ignored}}
__attribute__((arm_streaming))
int streaming_caller(void) {
  return normal_callee();
}

// expected-warning@+1 {{unknown attribute 'arm_streaming_compatible' ignored}}
__attribute__((arm_streaming_compatible))
int streaming_compatible_caller(void) {
  return normal_callee();
}

// expected-warning@+1 {{unknown attribute 'arm_locally_streaming' ignored}}
__attribute__((arm_locally_streaming))
int locally_streaming_caller(void) {
  return normal_callee();
}

// expected-warning@+1 {{unknown attribute 'arm_shared_za' ignored}}
__attribute__((arm_shared_za))
int shared_za_caller(void) {
  return normal_callee();
}

// expected-warning@+1 {{unknown attribute 'arm_preserves_za' ignored}}
__attribute__((arm_preserves_za))
int preserves_za_caller(void) {
  return normal_callee();
}

// expected-warning@+1 {{unknown attribute 'arm_new_za' ignored}}
__attribute__((arm_new_za))
int new_za_caller(void) {
  return normal_callee();
}
