// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -fsyntax-only -verify %s

#include <arm_sme.h>

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_hor_za8_bad_tile(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 1 is outside the valid range [0, 0]}}
  svld1_hor_za8(1, slice_base, 0, pg, base);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_hor_za16_bad_tile(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  svld1_hor_za16(2, slice_base, 0, pg, base);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_hor_za32_bad_tile(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svld1_hor_za32(4, slice_base, 0, pg, base);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_ver_za64_bad_tile(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svld1_ver_za64(8, slice_base, 0, pg, base);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_ver_za128_bad_tile(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  svld1_ver_za128(16, slice_base, 0, pg, base);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_hor_za8_slice_offset(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  svld1_hor_za8(0, slice_base, 16, pg, base);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_ver_za16_slice_offset(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svld1_ver_za16(0, slice_base, 8, pg, base);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_hor_za32_slice_offset(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svld1_hor_za32(0, slice_base, 4, pg, base);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_ver_za64_slice_offset(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 2 is outside the valid range [0, 1]}}
  svld1_ver_za64(0, slice_base, 2, pg, base);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svld1_ver_za128_slice_offset(uint32_t slice_base, svbool_t pg, const int8_t *base) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 0]}}
  svld1_ver_za128(0, slice_base, 16, pg, base);
}

__attribute__((arm_streaming_compatible, arm_shared_za))
void test_svldr_vnum_za(uint32_t slice_base, const uint8_t *base) {
  // expected-error@+1 {{argument value 16 is outside the valid range [0, 15]}}
  svldr_vnum_za(slice_base, 16, base);
}
