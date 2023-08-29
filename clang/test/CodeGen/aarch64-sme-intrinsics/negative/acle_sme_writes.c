// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -fsyntax-only -verify %s

#include <arm_sme.h>

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_hor_za8_bad_tile(uint32_t slice_base, svbool_t pg, svint8_t zn) {
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  svwrite_hor_za8_s8_m(1, slice_base, 0, pg, zn);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_ver_za16_bad_tile(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  svwrite_ver_za16_u16_m(2, slice_base, 0, pg, zn);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_ver_za32_bad_tile(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svwrite_ver_za32_f32_m(4, slice_base, 0, pg, zn);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_hor_za64_bad_tile(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svwrite_hor_za64_s64_m(8, slice_base, 0, pg, zn);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_hor_za128_bad_tile(uint32_t slice_base, svbool_t pg, svbfloat16_t zn) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  svwrite_hor_za128_bf16_m(16, slice_base, 0, pg, zn);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_hor_za8_bad_slice(uint32_t slice_base, svbool_t pg, svint8_t zn) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  svwrite_hor_za8_s8_m(0, slice_base, 16, pg, zn);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_ver_za16_bad_slice(uint32_t slice_base, svbool_t pg, svuint16_t zn) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svwrite_ver_za16_u16_m(1, slice_base, 8, pg, zn);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_ver_za32_bad_slice(uint32_t slice_base, svbool_t pg, svfloat32_t zn) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svwrite_ver_za32_f32_m(3, slice_base, 4, pg, zn);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_hor_za64_bad_slice(uint32_t slice_base, svbool_t pg, svint64_t zn) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  svwrite_hor_za64_s64_m(7, slice_base, 2, pg, zn);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svwrite_hor_za128_bad_slice(svint32_t zd, svbool_t pg, uint32_t slice_base) {
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  svwrite_hor_za128_s32_m(15, slice_base, 1, pg, zd);
}
