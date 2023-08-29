// RUN: %clang_cc1 -triple aarch64-none-linux-gnu \
// RUN:    -target-feature +sve2 -target-feature +sme2 -target-feature +sme-i16i64 -target-feature +sme-f64f64 -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include <arm_sme.h>

__attribute__((arm_streaming, arm_shared_za))
void test_multivector_read(uint32_t base) {

  // Test Tile Range
  svread_hor_za8_u8_vg2(1, base, 0); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svread_ver_za8_u8_vg2(1, base, 0); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svread_hor_za8_u8_vg4(1, base, 0); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svread_ver_za8_u8_vg4(1, base, 0); // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svread_hor_za16_u16_vg2(2, base, 0); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svread_ver_za16_u16_vg2(2, base, 0); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svread_hor_za16_u16_vg4(2, base, 0); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svread_ver_za16_u16_vg4(2, base, 0); // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  svread_hor_za32_u32_vg2(4, base, 0); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svread_ver_za32_u32_vg2(4, base, 0); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svread_hor_za32_u32_vg4(4, base, 0); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svread_ver_za32_u32_vg4(4, base, 0); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svread_hor_za64_u64_vg2(8, base, 0); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svread_ver_za64_u64_vg2(8, base, 0); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svread_hor_za64_u64_vg4(8, base, 0); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svread_ver_za64_u64_vg4(8, base, 0); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Test Offset Range
  svread_hor_za8_u8_vg2(0, base, 13); // expected-error {{argument should be a multiple of 2}}
  svread_hor_za8_u8_vg2(0, base, 16); // expected-error {{argument value 16 is outside the valid range [0, 14]}}
  svread_ver_za8_u8_vg4(0, base, 11); // expected-error {{argument should be a multiple of 4}}
  svread_ver_za8_u8_vg4(0, base, 16); // expected-error {{argument value 16 is outside the valid range [0, 12]}}

  svread_hor_za16_u16_vg2(0, base, 5); // expected-error {{argument should be a multiple of 2}}
  svread_hor_za16_u16_vg2(0, base, 8); // expected-error {{argument value 8 is outside the valid range [0, 6]}}
  svread_ver_za16_u16_vg2(0, base, 5); // expected-error {{argument should be a multiple of 2}}
  svread_ver_za16_u16_vg2(0, base, 8); // expected-error {{argument value 8 is outside the valid range [0, 6]}}

  svread_hor_za16_u16_vg4(0, base, 3); // expected-error {{argument should be a multiple of 4}}
  svread_hor_za16_u16_vg4(0, base, 8); // expected-error {{argument value 8 is outside the valid range [0, 4]}}
  svread_ver_za16_u16_vg4(0, base, 3); // expected-error {{argument should be a multiple of 4}}
  svread_ver_za16_u16_vg4(0, base, 8); // expected-error {{argument value 8 is outside the valid range [0, 4]}}

  svread_hor_za32_u32_vg2(0, base, 1); // expected-error {{argument should be a multiple of 2}}
  svread_hor_za32_u32_vg2(0, base, 4); // expected-error {{argument value 4 is outside the valid range [0, 2]}}
  svread_ver_za32_u32_vg2(0, base, 1); // expected-error {{argument should be a multiple of 2}}
  svread_ver_za32_u32_vg2(0, base, 4); // expected-error {{argument value 4 is outside the valid range [0, 2]}}

  svread_hor_za32_u32_vg4(0, base, 4); // expected-error {{argument value 4 is outside the valid range [0, 0]}}
  svread_hor_za64_u64_vg2(0, base, 2); // expected-error {{argument value 2 is outside the valid range [0, 0]}}
  svread_hor_za64_u64_vg4(0, base, 4); // expected-error {{argument value 4 is outside the valid range [0, 0]}}

  svread_za64_u64_vg1x2(base, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svread_za64_u64_vg1x4(base, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

__attribute__((arm_streaming, arm_shared_za))
void test_multivector_write(uint32_t base, svuint8x2_t v8x2, svuint8x4_t v8x4,
                            svuint16x2_t v16x2, svuint16x4_t v16x4,
                            svuint32x2_t v32x2, svuint32x4_t v32x4,
                            svuint64x2_t v64x2, svuint64x4_t v64x4) {

  // Test Tile Range
  svwrite_hor_za8_u8_vg2(1, base, 0, v8x2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svwrite_ver_za8_u8_vg2(1, base, 0, v8x2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svwrite_hor_za8_u8_vg4(1, base, 0, v8x4); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svwrite_ver_za8_u8_vg4(1, base, 0, v8x4); // expected-error {{argument value 1 is outside the valid range [0, 0]}}

  svwrite_hor_za16_u16_vg2(2, base, 0, v16x2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svwrite_ver_za16_u16_vg2(2, base, 0, v16x2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svwrite_hor_za16_u16_vg4(2, base, 0, v16x4); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svwrite_ver_za16_u16_vg4(2, base, 0, v16x4); // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  svwrite_hor_za32_u32_vg2(4, base, 0, v32x2); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svwrite_ver_za32_u32_vg2(4, base, 0, v32x2); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svwrite_hor_za32_u32_vg4(4, base, 0, v32x4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svwrite_ver_za32_u32_vg4(4, base, 0, v32x4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  svwrite_hor_za64_u64_vg2(8, base, 0, v64x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svwrite_ver_za64_u64_vg2(8, base, 0, v64x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svwrite_hor_za64_u64_vg4(8, base, 0, v64x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svwrite_ver_za64_u64_vg4(8, base, 0, v64x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Test Offset Range
  svwrite_hor_za8_u8_vg2(0, base, 13, v8x2); // expected-error {{argument should be a multiple of 2}}
  svwrite_hor_za8_u8_vg2(0, base, 16, v8x2); // expected-error {{argument value 16 is outside the valid range [0, 14]}}
  svwrite_ver_za8_u8_vg4(0, base, 11, v8x4); // expected-error {{argument should be a multiple of 4}}
  svwrite_ver_za8_u8_vg4(0, base, 16, v8x4); // expected-error {{argument value 16 is outside the valid range [0, 12]}}

  svwrite_hor_za16_u16_vg2(0, base, 5, v16x2); // expected-error {{argument should be a multiple of 2}}
  svwrite_hor_za16_u16_vg2(0, base, 8, v16x2); // expected-error {{argument value 8 is outside the valid range [0, 6]}}
  svwrite_ver_za16_u16_vg2(0, base, 5, v16x2); // expected-error {{argument should be a multiple of 2}}
  svwrite_ver_za16_u16_vg2(0, base, 8, v16x2); // expected-error {{argument value 8 is outside the valid range [0, 6]}}

  svwrite_hor_za16_u16_vg4(0, base, 3, v16x4); // expected-error {{argument should be a multiple of 4}}
  svwrite_hor_za16_u16_vg4(0, base, 8, v16x4); // expected-error {{argument value 8 is outside the valid range [0, 4]}}
  svwrite_ver_za16_u16_vg4(0, base, 3, v16x4); // expected-error {{argument should be a multiple of 4}}
  svwrite_ver_za16_u16_vg4(0, base, 8, v16x4); // expected-error {{argument value 8 is outside the valid range [0, 4]}}

  svwrite_hor_za32_u32_vg2(0, base, 1, v32x2); // expected-error {{argument should be a multiple of 2}}
  svwrite_hor_za32_u32_vg2(0, base, 4, v32x2); // expected-error {{argument value 4 is outside the valid range [0, 2]}}
  svwrite_ver_za32_u32_vg2(0, base, 1, v32x2); // expected-error {{argument should be a multiple of 2}}
  svwrite_ver_za32_u32_vg2(0, base, 4, v32x2); // expected-error {{argument value 4 is outside the valid range [0, 2]}}

  svwrite_hor_za32_u32_vg4(0, base, 4, v32x4); // expected-error {{argument value 4 is outside the valid range [0, 0]}}
  svwrite_hor_za64_u64_vg2(0, base, 2, v64x2); // expected-error {{argument value 2 is outside the valid range [0, 0]}}
  svwrite_hor_za64_u64_vg4(0, base, 4, v64x4); // expected-error {{argument value 4 is outside the valid range [0, 0]}}

  svwrite_za64_u64_vg1x2(base, 8, v64x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svwrite_za64_u64_vg1x4(base, 8, v64x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

__attribute__((arm_streaming, arm_shared_za))
void test_multiply_add_sub_long(uint32_t base, svint8_t s8, svuint8_t u8,
                                svint16_t s16, svuint16_t u16, svint8x2_t s8x2,
                                svuint8x2_t u8x2, svint16x2_t s16x2, svuint16x2_t u16x2,
                                svint8x4_t s8x4, svuint8x4_t u8x4, svint16x4_t s16x4, svuint16x4_t u16x4) {
  svmla_single_za32_s8_vg4x1(base, 13, s8, s8);    // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmla_single_za32_u8_vg4x1(base, 13, u8, u8);    // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmla_single_za64_s16_vg4x1(base, 13, s16, s16); // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmla_single_za64_u16_vg4x1(base, 13, u16, u16); // expected-error {{argument value 13 is outside the valid range [0, 12]}}

  svmla_single_za32_s8_vg4x2(base, 5, s8x2, s8);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_single_za32_u8_vg4x2(base, 5, u8x2, u8);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_single_za64_s16_vg4x2(base, 5, s16x2, s16); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_single_za64_u16_vg4x2(base, 5, u16x2, u16); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmla_single_za32_s8_vg4x4(base, 5, s8x4, s8);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_single_za32_u8_vg4x4(base, 5, u8x4, u8);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_single_za64_s16_vg4x4(base, 5, s16x4, s16); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_single_za64_u16_vg4x4(base, 5, u16x4, u16); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmls_single_za32_s8_vg4x1(base, 13, s8, s8);    // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmls_single_za32_u8_vg4x1(base, 13, u8, u8);    // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmls_single_za64_s16_vg4x1(base, 13, s16, s16); // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmls_single_za64_u16_vg4x1(base, 13, u16, u16); // expected-error {{argument value 13 is outside the valid range [0, 12]}}

  svmls_single_za32_s8_vg4x2(base, 5, s8x2, s8);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_single_za32_u8_vg4x2(base, 5, u8x2, u8);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_single_za64_s16_vg4x2(base, 5, s16x2, s16); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_single_za64_u16_vg4x2(base, 5, u16x2, u16); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmls_single_za32_s8_vg4x4(base, 5, s8x4, s8);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_single_za32_u8_vg4x4(base, 5, u8x4, u8);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_single_za64_s16_vg4x4(base, 5, s16x4, s16); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_single_za64_u16_vg4x4(base, 5, u16x4, u16); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svsumla_single_za32_u8_vg4x2(base, 5, s8x2, u8); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svsumla_single_za32_u8_vg4x4(base, 5, s8x4, u8); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svusmla_single_za32_s8_vg4x1(base, 13, u8, s8);  // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svusmla_single_za32_s8_vg4x2(base, 5, u8x2, s8); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svusmla_single_za32_s8_vg4x4(base, 5, u8x4, s8); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmla_za32_s8_vg4x2(base, 5, s8x2, s8x2);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_za32_u8_vg4x2(base, 5, u8x2, u8x2);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_za64_s16_vg4x2(base, 5, s16x2, s16x2); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_za64_u16_vg4x2(base, 5, u16x2, u16x2); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmla_za32_s8_vg4x4(base, 5, s8x4, s8x4);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_za32_u8_vg4x4(base, 5, u8x4, u8x4);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_za64_s16_vg4x4(base, 5, s16x4, s16x4); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_za64_u16_vg4x4(base, 5, u16x4, u16x4); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmls_za32_s8_vg4x2(base, 5, s8x2, s8x2);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_za32_u8_vg4x2(base, 5, u8x2, u8x2);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_za64_s16_vg4x2(base, 5, s16x2, s16x2); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_za64_u16_vg4x2(base, 5, u16x2, u16x2); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmls_za32_s8_vg4x4(base, 5, s8x4, s8x4);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_za32_u8_vg4x4(base, 5, u8x4, u8x4);    // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_za64_s16_vg4x4(base, 5, s16x4, s16x4); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_za64_u16_vg4x4(base, 5, u16x4, u16x4); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svusmla_za32_s8_vg4x2(base, 5, u8x2, s8x2); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svusmla_za32_s8_vg4x4(base, 5, u8x4, s8x4); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmla_lane_za32_s8_vg4x1(base, 13, s8, s8, 15);   // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmla_lane_za32_u8_vg4x1(base, 13, u8, u8, 15);   // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmla_lane_za64_s16_vg4x1(base, 13, s16, s16, 7); // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmla_lane_za64_u16_vg4x1(base, 13, u16, u16, 7); // expected-error {{argument value 13 is outside the valid range [0, 12]}}

  svmla_lane_za32_s8_vg4x2(base, 5, s8x2, s8, 15);   // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_lane_za32_u8_vg4x2(base, 5, u8x2, u8, 15);   // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_lane_za64_s16_vg4x2(base, 5, s16x2, s16, 7); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_lane_za64_u16_vg4x2(base, 5, u16x2, u16, 7); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmla_lane_za32_s8_vg4x4(base, 5, s8x4, s8, 15);   // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_lane_za32_u8_vg4x4(base, 5, u8x4, u8, 15);   // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_lane_za64_s16_vg4x4(base, 5, s16x4, s16, 7); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmla_lane_za64_u16_vg4x4(base, 5, u16x4, u16, 7); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmla_lane_za32_s8_vg4x1(base, 12, s8, s8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmla_lane_za32_u8_vg4x1(base, 12, u8, u8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmla_lane_za64_s16_vg4x1(base, 12, s16, s16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svmla_lane_za64_u16_vg4x1(base, 12, u16, u16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  svmla_lane_za32_s8_vg4x2(base, 4, s8x2, s8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmla_lane_za32_u8_vg4x2(base, 4, u8x2, u8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmla_lane_za64_s16_vg4x2(base, 4, s16x2, s16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svmla_lane_za64_u16_vg4x2(base, 4, u16x2, u16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  svmla_lane_za32_s8_vg4x4(base, 4, s8x4, s8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmla_lane_za32_u8_vg4x4(base, 4, u8x4, u8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmla_lane_za64_s16_vg4x4(base, 4, s16x4, s16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svmla_lane_za64_u16_vg4x4(base, 4, u16x4, u16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  svmls_lane_za32_s8_vg4x1(base, 13, s8, s8, 15);   // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmls_lane_za32_u8_vg4x1(base, 13, u8, u8, 15);   // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmls_lane_za64_s16_vg4x1(base, 13, s16, s16, 7); // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svmls_lane_za64_u16_vg4x1(base, 13, u16, u16, 7); // expected-error {{argument value 13 is outside the valid range [0, 12]}}

  svmls_lane_za32_s8_vg4x2(base, 5, s8x2, s8, 15);   // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_lane_za32_u8_vg4x2(base, 5, u8x2, u8, 15);   // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_lane_za64_s16_vg4x2(base, 5, s16x2, s16, 7); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_lane_za64_u16_vg4x2(base, 5, u16x2, u16, 7); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmls_lane_za32_s8_vg4x4(base, 5, s8x4, s8, 15);   // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_lane_za32_u8_vg4x4(base, 5, u8x4, u8, 15);   // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_lane_za64_s16_vg4x4(base, 5, s16x4, s16, 7); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svmls_lane_za64_u16_vg4x4(base, 5, u16x4, u16, 7); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svmls_lane_za32_s8_vg4x1(base, 12, s8, s8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmls_lane_za32_u8_vg4x1(base, 12, u8, u8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmls_lane_za64_s16_vg4x1(base, 12, s16, s16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svmls_lane_za64_u16_vg4x1(base, 12, u16, u16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  svmls_lane_za32_s8_vg4x2(base, 4, s8x2, s8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmls_lane_za32_u8_vg4x2(base, 4, u8x2, u8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmls_lane_za64_s16_vg4x2(base, 4, s16x2, s16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svmls_lane_za64_u16_vg4x2(base, 4, u16x2, u16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  svmls_lane_za32_s8_vg4x4(base, 4, s8x4, s8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmls_lane_za32_u8_vg4x4(base, 4, u8x4, u8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svmls_lane_za64_s16_vg4x4(base, 4, s16x4, s16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svmls_lane_za64_u16_vg4x4(base, 4, u16x4, u16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  svsumla_lane_za32_s8_vg4x1(base, 13, s8, u8, 15);  // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svsumla_lane_za32_s8_vg4x2(base, 5, s8x2, u8, 15); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svsumla_lane_za32_s8_vg4x4(base, 5, s8x4, u8, 15); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svsumla_lane_za32_s8_vg4x1(base, 12, s8, u8, 16);  // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svsumla_lane_za32_s8_vg4x2(base, 4, s8x2, u8, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svsumla_lane_za32_s8_vg4x4(base, 4, s8x4, u8, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}

  svusmla_lane_za32_u8_vg4x1(base, 13, u8, s8, 15);  // expected-error {{argument value 13 is outside the valid range [0, 12]}}
  svusmla_lane_za32_u8_vg4x2(base, 5, u8x2, s8, 15); // expected-error {{argument value 5 is outside the valid range [0, 4]}}
  svusmla_lane_za32_u8_vg4x4(base, 5, u8x4, s8, 15); // expected-error {{argument value 5 is outside the valid range [0, 4]}}

  svusmla_lane_za32_u8_vg4x1(base, 12, u8, s8, 16);  // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svusmla_lane_za32_u8_vg4x2(base, 4, u8x2, s8, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
  svusmla_lane_za32_u8_vg4x4(base, 4, u8x4, s8, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
}

__attribute__((arm_streaming, arm_shared_za))
void test_vertical_dot_product(uint32_t base, svint16x2_t s16x2, svuint16x2_t u16x2,
                               svint8x4_t s8x4, svuint8x4_t u8x4,
                               svint16x4_t s16x4, svuint16x4_t u16x4,
                               svfloat16x2_t f16x2, svbfloat16x2_t bf16x2,
                               svint16_t s16, svuint16_t u16,
                               svint8_t s8, svuint8_t u8,
                               svfloat16_t f16, svbfloat16_t b16) {
  // Test slice offset values.
  svvdot_lane_za32_s16_vg1x2(base, 8, s16x2, s16, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svvdot_lane_za32_u16_vg1x2(base, 8, u16x2, u16, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svvdot_lane_za32_s8_vg1x4(base, 8, s8x4, s8, 0);      // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svvdot_lane_za32_u8_vg1x4(base, 8, u8x4, u8, 0);      // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svvdot_lane_za64_s16_vg1x4(base, 8, s16x4, s16, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svvdot_lane_za64_u16_vg1x4(base, 8, u16x4, u16, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svvdot_lane_za32_f16_vg1x2(base, 8, f16x2, f16, 0);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svvdot_lane_za32_bf16_vg1x2(base, 8, bf16x2, b16, 0); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svsuvdot_lane_za32_s8_vg1x4(base, 8, s8x4, s8, 0);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svusvdot_lane_za32_u8_vg1x4(base, 8, u8x4, u8, 0);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Test lane indices.
  svvdot_lane_za32_s16_vg1x2(base, 7, s16x2, s16, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svvdot_lane_za32_u16_vg1x2(base, 7, u16x2, u16, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svvdot_lane_za32_s8_vg1x4(base, 7, s8x4, s8, 4);      // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svvdot_lane_za32_u8_vg1x4(base, 7, u8x4, u8, 4);      // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svvdot_lane_za64_s16_vg1x4(base, 7, s16x4, s16, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svvdot_lane_za64_u16_vg1x4(base, 7, u16x4, u16, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svvdot_lane_za32_f16_vg1x2(base, 7, f16x2, f16, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svvdot_lane_za32_bf16_vg1x2(base, 7, bf16x2, b16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svsuvdot_lane_za32_s8_vg1x4(base, 7, s8x4, s8, 4);    // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svusvdot_lane_za32_u8_vg1x4(base, 7, u8x4, u8, 4);    // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

__attribute__((arm_streaming, arm_shared_za))
void test_fdot_za32_bad_slice(uint32_t slice_base, svfloat16_t z_f16,
                              svfloat16x2_t z_f16x2, svfloat16x4_t z_f16x4,
                              svbfloat16_t z_bf16, svbfloat16x2_t z_bf16x2,
                              svbfloat16x4_t z_bf16x4) {

  // 16-bit float
  svdot_za32_f16_vg1x2(slice_base, 8, z_f16x2, z_f16x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za32_f16_vg1x4(slice_base, 8, z_f16x4, z_f16x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_f16_vg1x2(slice_base, 8, z_f16x2, z_f16);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_f16_vg1x4(slice_base, 8, z_f16x4, z_f16);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_f16_vg1x2(slice_base, 8, z_f16x2, z_f16, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_f16_vg1x4(slice_base, 8, z_f16x4, z_f16, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // 16-bit binary float
  svdot_za32_bf16_vg1x2(slice_base, 8, z_bf16x2, z_bf16x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za32_bf16_vg1x4(slice_base, 8, z_bf16x4, z_bf16x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_bf16_vg1x2(slice_base, 8, z_bf16x2, z_bf16);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_bf16_vg1x4(slice_base, 8, z_bf16x4, z_bf16);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_bf16_vg1x2(slice_base, 8, z_bf16x2, z_bf16, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_bf16_vg1x4(slice_base, 8, z_bf16x4, z_bf16, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

__attribute__((arm_streaming, arm_shared_za))
void test_fdot_za32_bad_lane(uint32_t slice_base, svfloat16_t z_f16,
                             svfloat16x2_t z_f16x2, svfloat16x4_t z_f16x4,
                             svbfloat16_t z_bf16, svbfloat16x2_t z_bf16x2,
                             svbfloat16x4_t z_bf16x4) {
  // 16-bit float
  svdot_lane_za32_f16_vg1x2(slice_base, 7, z_f16x2, z_f16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_f16_vg1x4(slice_base, 7, z_f16x4, z_f16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  // 16-bit binary float
  svdot_lane_za32_bf16_vg1x2(slice_base, 7, z_bf16x2, z_bf16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_bf16_vg1x4(slice_base, 7, z_bf16x4, z_bf16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

__attribute__((arm_streaming, arm_shared_za))
void test_svdot_multi_za32_bad_slice(uint32_t slice_base, svuint16_t z_u16,
                                     svuint16x2_t z_u16x2, svuint16x4_t z_u16x4,
                                     svint16_t z_s16, svint16x2_t z_s16x2,
                                     svint16x4_t z_s16x4, svuint8_t z_u8,
                                     svuint8x2_t z_u8x2, svuint8x4_t z_u8x4,
                                     svint8_t z_s8, svint8x2_t z_s8x2,
                                     svint8x4_t z_s8x4) {
  // Multi, multi (unsigned)
  svdot_za32_u16_vg1x2(slice_base, 8, z_u16x2, z_u16x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za32_u16_vg1x4(slice_base, 8, z_u16x4, z_u16x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za32_u8_vg1x2(slice_base, 8, z_u8x2, z_u8x2);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za32_u8_vg1x4(slice_base, 8, z_u8x4, z_u8x4);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za64_u16_vg1x2(slice_base, 8, z_u16x2, z_u16x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za64_u16_vg1x4(slice_base, 8, z_u16x4, z_u16x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, multi (signed)
  svdot_za32_s16_vg1x2(slice_base, 8, z_s16x2, z_s16x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za32_s16_vg1x4(slice_base, 8, z_s16x4, z_s16x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za32_s8_vg1x2(slice_base, 8, z_s8x2, z_s8x2);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za32_s8_vg1x4(slice_base, 8, z_s8x4, z_s8x4);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za64_s16_vg1x2(slice_base, 8, z_s16x2, z_s16x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_za64_s16_vg1x4(slice_base, 8, z_s16x4, z_s16x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, single (unsigned)
  svdot_single_za32_u16_vg1x2(slice_base, 8, z_u16x2, z_u16); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_u16_vg1x4(slice_base, 8, z_u16x4, z_u16); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_u8_vg1x2(slice_base, 8, z_u8x2, z_u8);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_u8_vg1x4(slice_base, 8, z_u8x4, z_u8);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za64_u16_vg1x2(slice_base, 8, z_u16x2, z_u16); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za64_u16_vg1x4(slice_base, 8, z_u16x4, z_u16); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, single (signed)
  svdot_single_za32_s16_vg1x2(slice_base, 8, z_s16x2, z_s16); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_s16_vg1x4(slice_base, 8, z_s16x4, z_s16); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_s8_vg1x2(slice_base, 8, z_s8x2, z_s8);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za32_s8_vg1x4(slice_base, 8, z_s8x4, z_s8);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za64_s16_vg1x2(slice_base, 8, z_s16x2, z_s16); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_single_za64_s16_vg1x4(slice_base, 8, z_s16x4, z_s16); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, indexed (unsigned)
  svdot_lane_za32_u16_vg1x2(slice_base, 8, z_u16x2, z_u16, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_u16_vg1x4(slice_base, 8, z_u16x4, z_u16, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_u8_vg1x2(slice_base, 8, z_u8x2, z_u8, 3);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_u8_vg1x4(slice_base, 8, z_u8x4, z_u8, 3);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za64_u16_vg1x2(slice_base, 8, z_u16x2, z_u16, 1); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za64_u16_vg1x4(slice_base, 8, z_u16x4, z_u16, 1); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, indexed (signed)
  svdot_lane_za32_s16_vg1x2(slice_base, 8, z_s16x2, z_s16, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_s16_vg1x4(slice_base, 8, z_s16x4, z_s16, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_s8_vg1x2(slice_base, 8, z_s8x2, z_s8, 3);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za32_s8_vg1x4(slice_base, 8, z_s8x4, z_s8, 3);    // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za64_s16_vg1x2(slice_base, 8, z_s16x2, z_s16, 1); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svdot_lane_za64_s16_vg1x4(slice_base, 8, z_s16x4, z_s16, 1); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, multi (unsigned by signed)
  svusdot_za32_u8_vg1x2(slice_base, 8, z_u8x2, z_s8x2); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svusdot_za32_u8_vg1x4(slice_base, 8, z_u8x4, z_s8x4); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, single (unsigned by signed)
  svusdot_single_za32_u8_vg1x2(slice_base, 8, z_u8x2, z_s8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svusdot_single_za32_u8_vg1x4(slice_base, 8, z_u8x4, z_s8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, indexed (unsigned by signed)
  svusdot_lane_za32_u8_vg1x2(slice_base, 8, z_u8x2, z_s8, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svusdot_lane_za32_u8_vg1x4(slice_base, 8, z_u8x4, z_s8, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, single (signed by unsigned)
  svsudot_single_za32_s8_vg1x2(slice_base, 8, z_s8x2, z_u8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svsudot_single_za32_s8_vg1x4(slice_base, 8, z_s8x4, z_u8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Multi, indexed (unsigned by signed)
  svsudot_lane_za32_s8_vg1x2(slice_base, 8, z_s8x2, z_u8, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svsudot_lane_za32_s8_vg1x4(slice_base, 8, z_s8x4, z_u8, 3); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}


__attribute__((arm_streaming, arm_shared_za))
void test_svdot_multi_za32_bad_lane(uint32_t slice_base, svuint16_t z_u16,
                                    svuint16x2_t z_u16x2, svuint16x4_t z_u16x4,
                                    svint16_t z_s16, svint16x2_t z_s16x2,
                                    svint16x4_t z_s16x4, svuint8_t z_u8,
                                    svuint8x2_t z_u8x2, svuint8x4_t z_u8x4,
                                    svint8_t z_s8, svint8x2_t z_s8x2,
                                    svint8x4_t z_s8x4) {
  // Multi, indexed (unsigned)
  svdot_lane_za32_u16_vg1x2(slice_base, 7, z_u16x2, z_u16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_u16_vg1x4(slice_base, 7, z_u16x4, z_u16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_u8_vg1x2(slice_base, 7, z_u8x2, z_u8, 4);    // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_u8_vg1x4(slice_base, 7, z_u8x4, z_u8, 4);    // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za64_u16_vg1x2(slice_base, 7, z_u16x2, z_u16, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svdot_lane_za64_u16_vg1x4(slice_base, 7, z_u16x4, z_u16, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  // Multi, indexed (signed)
  svdot_lane_za32_s16_vg1x2(slice_base, 7, z_s16x2, z_s16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_s16_vg1x4(slice_base, 7, z_s16x4, z_s16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_s8_vg1x2(slice_base, 7, z_s8x2, z_s8, 4);    // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za32_s8_vg1x4(slice_base, 7, z_s8x4, z_s8, 4);    // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svdot_lane_za64_s16_vg1x2(slice_base, 7, z_s16x2, z_s16, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}
  svdot_lane_za64_s16_vg1x4(slice_base, 7, z_s16x4, z_s16, 2); // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  // Multi, indexed (unsigned by signed)
  svusdot_lane_za32_u8_vg1x2(slice_base, 7, z_u8x2, z_s8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svusdot_lane_za32_u8_vg1x4(slice_base, 7, z_u8x4, z_s8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  // Multi, indexed (unsigned by signed)
  svsudot_lane_za32_s8_vg1x2(slice_base, 7, z_s8x2, z_u8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
  svsudot_lane_za32_s8_vg1x4(slice_base, 7, z_s8x4, z_u8, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

__attribute__((arm_streaming_compatible))
void test_bfmlslb_bad_lane(svfloat32_t zda, svbfloat16_t zn, svbfloat16_t zm) {
  svbfmlslb_lane_f32(zda, zn, zm, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
  svbfmlslt_lane_f32(zda, zn, zm, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

__attribute__((arm_streaming_compatible, arm_shared_za, arm_preserves_za))
void test_ldr_str_zt(const void *const_base, void *base) {
  svldr_zt(1, const_base); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  svstr_zt(1, base);       // expected-error {{argument value 1 is outside the valid range [0, 0]}}
}

__attribute__((arm_streaming_compatible, arm_shared_za, arm_preserves_za))
void test_zero_zt() {
  // Test Reg Offset
  svzero_zt(1); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
}

__attribute__((arm_streaming, arm_shared_za, arm_preserves_za))
void test_svluti2_lane_zt(svuint8_t zn_u8, svuint16_t zn_u16, svuint32_t zn_u32) {
  // Test Reg Offset
  svluti2_lane_zt_u8(1, zn_u8, 2);    // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u8(0, zn_u8, 16);   // expected-error {{argument value 16 is outside the valid range [0, 15]}}

  // Test Reg Offset
  svluti2_lane_zt_u16(1, zn_u16, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u16(0, zn_u16, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}

  // Test Reg Offset
  svluti2_lane_zt_u32(1, zn_u32, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u32(0, zn_u32, 16); // expected-error {{argument value 16 is outside the valid range [0, 15]}}
}

__attribute__((arm_streaming, arm_shared_za, arm_preserves_za))
void test_svluti4_lane_zt(svuint8_t zn_u8, svuint16_t zn_u16, svuint32_t zn_u32) {
  // Test Reg Offset
  svluti4_lane_zt_u8(1, zn_u8, 2);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u8(0, zn_u8, 8);  // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Test Reg Offset
  svluti4_lane_zt_u16(1, zn_u16, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u16(0, zn_u16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Test Reg Offset
  svluti4_lane_zt_u32(1, zn_u32, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u32(0, zn_u32, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

__attribute__((arm_streaming, arm_shared_za, arm_preserves_za))
void test_svluti2_lane_zt_x2(svuint8_t zn_u8, svuint16_t zn_u16, svuint32_t zn_u32) {
  // Test Reg Offset
  svluti2_lane_zt_u8_x2(1, zn_u8, 2);    // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u8_x2(0, zn_u8, 8);   // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Test Reg Offset
  svluti2_lane_zt_u16_x2(1, zn_u16, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u16_x2(0, zn_u16, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}

  // Test Reg Offset
  svluti2_lane_zt_u32_x2(1, zn_u32, 2);  // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u32_x2(0, zn_u32, 8); // expected-error {{argument value 8 is outside the valid range [0, 7]}}
}

__attribute__((arm_streaming, arm_shared_za, arm_preserves_za))
void test_svluti4_lane_zt_x2(svuint8_t zn_u8, svuint16_t zn_u16, svuint32_t zn_u32) {
  // Test Reg Offset
  svluti4_lane_zt_u8_x2(1, zn_u8, 2);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u8_x2(0, zn_u8, 4);  // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  // Test Reg Offset
  svluti4_lane_zt_u16_x2(1, zn_u16, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u16_x2(0, zn_u16, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  // Test Reg Offset
  svluti4_lane_zt_u32_x2(1, zn_u32, 2); // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u32_x2(0, zn_u32, 4); // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

__attribute__((arm_streaming, arm_shared_za, arm_preserves_za))
void test_svluti2_lane_zt_x4(svuint8_t zn_u8, svuint16_t zn_u16, svuint32_t zn_u32) {
  // Test Reg Offset
  svluti2_lane_zt_u8_x4(1, zn_u8, 0);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u8_x4(0, zn_u8, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  // Test Reg Offset
  svluti2_lane_zt_u16_x4(1, zn_u16, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u16_x4(0, zn_u16, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}

  // Test Reg Offset
  svluti2_lane_zt_u32_x4(1, zn_u32, 3);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti2_lane_zt_u32_x4(0, zn_u32, 4);   // expected-error {{argument value 4 is outside the valid range [0, 3]}}
}

__attribute__((arm_streaming, arm_shared_za, arm_preserves_za))
void test_svluti4_lane_zt_x4(svuint8_t zn_u8, svuint16_t zn_u16, svuint32_t zn_u32) {
  // Test Reg Offset
  svluti4_lane_zt_u16_x4(1, zn_u16, 0);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u16_x4(0, zn_u16, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}

  // Test Reg Offset
  svluti4_lane_zt_u32_x4(1, zn_u32, 1);   // expected-error {{argument value 1 is outside the valid range [0, 0]}}
  // Test index value range
  svluti4_lane_zt_u32_x4(0, zn_u32, 2);   // expected-error {{argument value 2 is outside the valid range [0, 1]}}
}
