// RUN: %clang_cc1 -triple riscv64 -target-feature +f -target-feature +d \
// RUN:   -target-feature +v -target-feature +zfh -target-feature +zvfh \
// RUN:   -disable-O0-optnone -o - -fsyntax-only %s -verify 
// REQUIRES: riscv-registered-target

#include <riscv_vector.h>

void test_builtin_elementwise_abs(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                 vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_abs(vu8);
  // expected-error@-1 {{1st argument must be a signed integer or floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_abs(vu16);
  // expected-error@-1 {{1st argument must be a signed integer or floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_abs(vu32);
  // expected-error@-1 {{1st argument must be a signed integer or floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_abs(vu64);
  // expected-error@-1 {{1st argument must be a signed integer or floating point type}}
}

void test_builtin_elementwise_add_sat(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                 vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vfloat16m1_t res_vf16 = __builtin_elementwise_add_sat(vf16, vf16);
  // expected-error@-1 {{1st argument must be a vector of integers}}
  vfloat32m1_t res_vf32 = __builtin_elementwise_add_sat(vf32, vf32);
  // expected-error@-1 {{1st argument must be a vector of integers}}
  vfloat64m1_t res_vf64 = __builtin_elementwise_add_sat(vf64, vf64);
  // expected-error@-1 {{1st argument must be a vector of integers}}
}

void test_builtin_elementwise_sub_sat(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                 vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vfloat16m1_t res_vf16 = __builtin_elementwise_sub_sat(vf16, vf16);
  // expected-error@-1 {{1st argument must be a vector of integers}}
  vfloat32m1_t res_vf32 = __builtin_elementwise_sub_sat(vf32, vf32);
  // expected-error@-1 {{1st argument must be a vector of integers}}
  vfloat64m1_t res_vf64 = __builtin_elementwise_sub_sat(vf64, vf64);
  // expected-error@-1 {{1st argument must be a vector of integers}}
}

void test_builtin_elementwise_bitreverse(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vfloat16m1_t res_vf16 = __builtin_elementwise_bitreverse(vf16);
  // expected-error@-1 {{1st argument must be a vector of integers}}
  vfloat32m1_t res_vf32 = __builtin_elementwise_bitreverse(vf32);
  // expected-error@-1 {{1st argument must be a vector of integers}}
  vfloat64m1_t res_vf64 = __builtin_elementwise_bitreverse(vf64);
  // expected-error@-1 {{1st argument must be a vector of integers}}
}

void test_builtin_elementwise_ceil(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_ceil(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_ceil(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_ceil(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_ceil(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_ceil(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_ceil(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_ceil(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_ceil(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_cos(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_cos(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_cos(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_cos(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_cos(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_cos(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_cos(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_cos(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_cos(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_exp(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_exp(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_exp(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_exp(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_exp(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_exp(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_exp(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_exp(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_exp(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_exp2(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_exp2(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_exp2(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_exp2(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_exp2(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_exp2(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_exp2(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_exp2(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_exp2(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_floor(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_floor(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_floor(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_floor(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_floor(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_floor(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_floor(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_floor(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_floor(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_log(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_log(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_log(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_log(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_log(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_log(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_log(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_log(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_log(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_log10(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_log10(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_log10(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_log10(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_log10(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_log10(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_log10(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_log10(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_log10(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_log2(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_log2(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_log2(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_log2(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_log2(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_log2(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_log2(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_log2(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_log2(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_pow(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_pow(vu8, vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_pow(vu16, vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_pow(vu32, vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_pow(vu64, vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_pow(vi8, vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_pow(vi16, vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_pow(vi32, vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_pow(vi64, vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_roundeven(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_roundeven(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_roundeven(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_roundeven(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_roundeven(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_roundeven(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_roundeven(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_roundeven(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_roundeven(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_round(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_round(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_round(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_round(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_round(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_round(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_round(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_round(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_round(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_rint(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_rint(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_rint(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_rint(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_rint(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_rint(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_rint(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_rint(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_rint(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_nearbyint(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_nearbyint(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_nearbyint(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_nearbyint(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_nearbyint(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_nearbyint(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_nearbyint(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_nearbyint(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_nearbyint(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_sin(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_sin(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_sin(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_sin(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_sin(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_sin(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_sin(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_sin(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_sin(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_sqrt(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_sqrt(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_sqrt(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_sqrt(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_sqrt(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_sqrt(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_sqrt(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_sqrt(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_sqrt(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_trunc(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_trunc(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_trunc(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_trunc(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_trunc(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_trunc(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_trunc(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_trunc(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_trunc(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_canonicalize(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_canonicalize(vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_canonicalize(vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_canonicalize(vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_canonicalize(vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_canonicalize(vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_canonicalize(vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_canonicalize(vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_canonicalize(vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_copysign(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_copysign(vu8, vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_copysign(vu16, vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_copysign(vu32, vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_copysign(vu64, vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_copysign(vi8, vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_copysign(vi16, vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_copysign(vi32, vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_copysign(vi64, vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}

void test_builtin_elementwise_fma(vint8m1_t vi8, vint16m1_t vi16,
                                  vint32m1_t vi32, vint64m1_t vi64,
                                  vuint8m1_t vu8, vuint16m1_t vu16,
                                  vuint32m1_t vu32, vuint64m1_t vu64,
                                  vfloat16m1_t vf16, vfloat32m1_t vf32,
                                  vfloat64m1_t vf64) {
  vuint8m1_t res_vu8 = __builtin_elementwise_fma(vu8, vu8, vu8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vu16 = __builtin_elementwise_fma(vu16, vu16, vu16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vu32 = __builtin_elementwise_fma(vu32, vu32, vu32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vu64 = __builtin_elementwise_fma(vu64, vu64, vu64);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint8m1_t res_vi8 = __builtin_elementwise_fma(vi8, vi8, vi8);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint16m1_t res_vi16 = __builtin_elementwise_fma(vi16, vi16, vi16);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint32m1_t res_vi32 = __builtin_elementwise_fma(vi32, vi32, vi32);
  // expected-error@-1 {{1st argument must be a floating point type}}
  vuint64m1_t res_vi64 = __builtin_elementwise_fma(vi64, vi64, vi64);
  // expected-error@-1 {{1st argument must be a floating point type}}
}
