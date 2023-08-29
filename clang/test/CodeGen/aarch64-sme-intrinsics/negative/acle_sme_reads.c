// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -fsyntax-only -verify %s

#include <arm_sme.h>

__attribute__((arm_streaming, arm_shared_za))
void test_svread_hor_za8_bad_tile(svint8_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  svread_hor_za8_s8_m(zd, pg, 1, slice_base, 0);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svread_ver_za16_bad_tile(svuint16_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  svread_ver_za16_u16_m(zd, pg, 2, slice_base, 0);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svread_ver_za32_bad_tile(svfloat32_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svread_ver_za32_f32_m(zd, pg, 4, slice_base, 0);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svread_hor_za64_bad_tile(svint64_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svread_hor_za64_s64_m(zd, pg, 8, slice_base, 0);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svread_hor_za128_bad_tile(svbfloat16_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  svread_hor_za128_bf16_m(zd, pg, 16, slice_base, 0);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svread_hor_za8_bad_slice(svint8_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  svread_hor_za8_s8_m(zd, pg, 0, slice_base, 16);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svread_ver_za16_bad_slice(svuint16_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svread_ver_za16_u16_m(zd, pg, 1, slice_base, 8);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svread_ver_za32_bad_slice(svfloat32_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svread_ver_za32_f32_m(zd, pg, 3, slice_base, 4);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svread_hor_za64_bad_slice(svint64_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  svread_hor_za64_s64_m(zd, pg, 7, slice_base, 2);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svread_hor_za128_bad_slice(svint32_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  svread_hor_za128_s32_m(zd, pg, 15, slice_base, 1);
}
