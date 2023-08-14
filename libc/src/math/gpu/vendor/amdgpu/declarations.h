//===-- AMDGPU specific declarations for math support ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GPU_AMDGPU_DECLARATIONS_H
#define LLVM_LIBC_SRC_MATH_GPU_AMDGPU_DECLARATIONS_H

namespace __llvm_libc {

extern "C" {
double __ocml_acos_f64(double);
float __ocml_acos_f32(float);
double __ocml_acosh_f64(double);
float __ocml_acosh_f32(float);
double __ocml_asin_f64(double);
float __ocml_asin_f32(float);
double __ocml_asinh_f64(double);
float __ocml_asinh_f32(float);
double __ocml_atan_f64(double);
float __ocml_atan_f32(float);
double __ocml_atan2_f64(double, double);
float __ocml_atan2_f32(float, float);
double __ocml_atanh_f64(double);
float __ocml_atanh_f32(float);
double __ocml_ceil_f64(double);
float __ocml_ceil_f32(float);
double __ocml_copysign_f64(double, double);
float __ocml_copysign_f32(float, float);
double __ocml_cos_f64(double);
float __ocml_cos_f32(float);
double __ocml_cosh_f64(double);
float __ocml_cosh_f32(float);
double __ocml_erf_f64(double);
float __ocml_erf_f32(float);
double __ocml_exp_f64(double);
float __ocml_exp_f32(float);
double __ocml_exp10_f64(double);
float __ocml_exp10_f32(float);
double __ocml_exp2_f64(double);
float __ocml_exp2_f32(float);
double __ocml_expm1_f64(double);
float __ocml_expm1_f32(float);
double __ocml_fabs_f64(double);
float __ocml_fabs_f32(float);
double __ocml_fdim_f64(double, double);
float __ocml_fdim_f32(float, float);
double __ocml_fma_f64(double, double, double);
float __ocml_fma_f32(float, float, float);
double __ocml_fmax_f64(double, double);
float __ocml_fmax_f32(float, float);
double __ocml_fmin_f64(double, double);
float __ocml_fmin_f32(float, float);
double __ocml_floor_f64(double);
float __ocml_floor_f32(float);
double __ocml_fmod_f64(double, double);
float __ocml_fmod_f32(float, float);
double __ocml_frexp_f64(double, int *);
float __ocml_frexp_f32(float, int *);
double __ocml_hypot_f64(double, double);
float __ocml_hypot_f32(float, float);
int __ocml_ilogb_f64(double);
int __ocml_ilogb_f32(float);
double __ocml_ldexp_f64(double x, int exp);
float __ocml_ldexp_f32(float x, int exp);
long long __ocml_llrint_f64(double);
long long __ocml_llrint_f32(float);
double __ocml_log10_f64(double);
float __ocml_log10_f32(float);
double __ocml_log1p_f64(double);
float __ocml_log1p_f32(float);
double __ocml_log2_f64(double);
float __ocml_log2_f32(float);
double __ocml_logb_f64(double);
float __ocml_logb_f32(float);
double __ocml_log_f64(double);
float __ocml_log_f32(float);
long __ocml_lrint_f64(double);
long __ocml_lrint_f32(float);
long __ocml_lround_f64(double);
long __ocml_lround_f32(float);
double __ocml_modf_f64(double, double *);
float __ocml_modf_f32(float, float *);
double __ocml_nearbyint_f64(double);
float __ocml_nearbyint_f32(float);
double __ocml_nextafter_f64(double, double);
float __ocml_nextafter_f32(float, float);
double __ocml_pow_f64(double, double);
float __ocml_pow_f32(float, float);
double __ocml_remainder_f64(double, double);
float __ocml_remainder_f32(float, float);
double __ocml_remquo_f64(double, double, int *);
float __ocml_remquo_f32(float, float, int *);
double __ocml_rint_f64(double);
float __ocml_rint_f32(float);
double __ocml_round_f64(double);
float __ocml_round_f32(float);
double __ocml_scalb_f64(double, int);
float __ocml_scalb_f32(float, int);
double __ocml_sin_f64(double);
float __ocml_sin_f32(float);
double __ocml_sincos_f64(double, double *);
float __ocml_sincos_f32(float, float *);
double __ocml_sinh_f64(double);
float __ocml_sinh_f32(float);
double __ocml_sqrt_f64(double);
float __ocml_sqrt_f32(float);
double __ocml_tan_f64(double);
float __ocml_tan_f32(float);
double __ocml_tanh_f64(double);
float __ocml_tanh_f32(float);
double __ocml_tgamma_f64(double);
float __ocml_tgamma_f32(float);
double __ocml_trunc_f64(double);
float __ocml_trunc_f32(float);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_GPU_AMDGPU_DECLARATIONS_H
