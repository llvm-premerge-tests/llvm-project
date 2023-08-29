// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 \
// RUN:   -triple aarch64-none-linux-gnu -target-feature +sme-i16i64 -target-feature +sme-f64f64 -fsyntax-only -verify %s

#include <arm_sme.h>

// == MOPS / MOPA ==

__attribute__((arm_streaming, arm_shared_za))
void test_svmopa_za32_bf16(svbool_t pn, svbool_t pm, svbfloat16_t zn, svbfloat16_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmopa_za32_bf16_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmopa_za32_f16(svbool_t pn, svbool_t pm, svfloat16_t zn, svfloat16_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmopa_za32_f16_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmopa_za32_s8(svbool_t pn, svbool_t pm, svint8_t zn, svint8_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmopa_za32_s8_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmopa_za32_u8(svbool_t pn, svbool_t pm, svuint8_t zn, svuint8_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmopa_za32_u8_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmopa_za64_s16(svbool_t pn, svbool_t pm, svint16_t zn, svint16_t zm) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svmopa_za64_s16_m(8, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmopa_za64_u16(svbool_t pn, svbool_t pm, svuint16_t zn, svuint16_t zm) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svmopa_za64_u16_m(8, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmopa_za32_f32(svbool_t pn, svbool_t pm, svfloat32_t zn, svfloat32_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmopa_za32_f32_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmopa_za64_f64(svbool_t pn, svbool_t pm, svfloat64_t zn, svfloat64_t zm) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svmopa_za64_f64_m(8, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmops_za32_bf16(svbool_t pn, svbool_t pm, svbfloat16_t zn, svbfloat16_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmops_za32_bf16_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmops_za32_f16(svbool_t pn, svbool_t pm, svfloat16_t zn, svfloat16_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmops_za32_f16_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmops_za32_s8(svbool_t pn, svbool_t pm, svint8_t zn, svint8_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmops_za32_s8_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmops_za32_u8(svbool_t pn, svbool_t pm, svuint8_t zn, svuint8_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmops_za32_u8_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmops_za64_s16(svbool_t pn, svbool_t pm, svint16_t zn, svint16_t zm) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svmops_za64_s16_m(8, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmops_za64_u16(svbool_t pn, svbool_t pm, svuint16_t zn, svuint16_t zm) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svmops_za64_u16_m(8, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmops_za32_f32(svbool_t pn, svbool_t pm, svfloat32_t zn, svfloat32_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svmops_za32_f32_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svmops_za64_f64(svbool_t pn, svbool_t pm, svfloat64_t zn, svfloat64_t zm) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svmops_za64_f64_m(8, pn, pm, zn, zm);
}

// == MIXED SIGN MOPA / MOPS ==

__attribute__((arm_streaming, arm_shared_za))
void test_svsumops_za32_s8(svbool_t pn, svbool_t pm, svint8_t zn, svuint8_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svsumops_za32_s8_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svsumops_za64_s16(svbool_t pn, svbool_t pm, svint16_t zn, svuint16_t zm) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svsumops_za64_s16_m(8, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svusmops_za32_u8(svbool_t pn, svbool_t pm, svuint8_t zn, svint8_t zm) {
  // expected-error@+1 {{argument value 4 is outside the valid range [0, 3]}}
  svusmops_za32_u8_m(4, pn, pm, zn, zm);
}

__attribute__((arm_streaming, arm_shared_za))
void test_svusmops_za64_u16(svbool_t pn, svbool_t pm, svuint16_t zn, svint16_t zm) {
  // expected-error@+1 {{argument value 8 is outside the valid range [0, 7]}}
  svusmops_za64_u16_m(8, pn, pm, zn, zm);
}
