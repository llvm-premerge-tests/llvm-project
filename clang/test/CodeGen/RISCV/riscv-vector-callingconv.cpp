// REQUIRES: riscv-registered-target
// RUN: %clang_cc1 %s -triple riscv64 -target-feature +v -verify

__attribute__((riscv_vector_cc)) int var; // expected-warning {{'riscv_vector_cc' only applies to function types; type here is 'int'}}

__attribute__((riscv_vector_cc)) void func();
__attribute__((riscv_vector_cc(1))) void func_invalid(); // expected-error {{'riscv_vector_cc' attribute takes no arguments}}

void test_no_attribute(int); // expected-note {{previous declaration is here}}
void __attribute__((riscv_vector_cc)) test_no_attribute(int x) { } // expected-error {{function declared 'riscv_vector_cc' here was previously declared without calling convention}}

class test_cc {
  __attribute__((riscv_vector_cc)) void member_func();
};

void test_lambda() {
  __attribute__((riscv_vector_cc)) auto lambda = []() { // expected-warning {{'riscv_vector_cc' only applies to function types; type here is 'auto'}}
  };
}
