// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sve -fsyntax-only -verify -emit-llvm -o - %s
// REQUIRES: aarch64-registered-target

// Test that functions with the correct target attributes can use the correct SME intrinsics.

__attribute__((target("sme")))
void test_sme(__SVBool_t pg, void *ptr) {
  // expected-warning@+2 {{builtin call has undefined behaviour when called from a non-streaming function}}
  // expected-warning@+1 {{builtin call is not valid when calling from a function without active ZA state}}
  __builtin_sme_svld1_hor_za8(0, 0, 0, pg, ptr);
}

__attribute__((target("arch=armv8-a+sme")))
void test_arch_sme(__SVBool_t pg, void *ptr) {
  // expected-warning@+2 {{builtin call has undefined behaviour when called from a non-streaming function}}
  // expected-warning@+1 {{builtin call is not valid when calling from a function without active ZA state}}
  __builtin_sme_svld1_hor_vnum_za32(0, 0, 0, pg, ptr, 0);
}

__attribute__((target("+sme")))
void test_plus_sme(__SVBool_t pg, void *ptr) {
  // expected-warning@+2 {{builtin call has undefined behaviour when called from a non-streaming function}}
  // expected-warning@+1 {{builtin call is not valid when calling from a function without active ZA state}}
  __builtin_sme_svst1_ver_za16(0, 0, 0, pg, ptr);
}

void undefined(__SVBool_t pg, void *ptr) {
  // expected-warning@+2 {{builtin call has undefined behaviour when called from a non-streaming function}}
  // expected-warning@+1 {{builtin call is not valid when calling from a function without active ZA state}}
  __builtin_sme_svst1_ver_vnum_za64(0, 0, 0, pg, ptr, 0); // expected-error {{'__builtin_sme_svst1_ver_vnum_za64' needs target feature sme}}
}
