//===-- NVPTX specific declarations for math support ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_MATH_GPU_NVPTX_DECLARATIONS_H
#define LLVM_LIBC_SRC_MATH_GPU_NVPTX_DECLARATIONS_H

namespace __llvm_libc {

extern "C" {
double __nv_acos(double);
float __nv_acosf(float);
double __nv_acosh(double);
float __nv_acoshf(float);
double __nv_asin(double);
float __nv_asinf(float);
double __nv_asinh(double);
float __nv_asinhf(float);
double __nv_atan(double);
float __nv_atanf(float);
double __nv_atanh(double);
double __nv_atan2(double, double);
float __nv_atan2f(float, float);
float __nv_atanhf(float);
double __nv_ceil(double);
float __nv_ceilf(float);
double __nv_copysign(double, double);
float __nv_copysignf(float, float);
double __nv_cos(double);
float __nv_cosf(float);
double __nv_cosh(double);
float __nv_coshf(float);
double __nv_erf(double);
float __nv_erff(float);
double __nv_exp(double);
float __nv_expf(float);
double __nv_exp10(double);
float __nv_exp10f(float);
double __nv_exp2(double);
float __nv_exp2f(float);
double __nv_expm1(double);
float __nv_expm1f(float);
double __nv_fabs(double);
float __nv_fabsf(float);
double __nv_fdim(double, double);
float __nv_fdimf(float, float);
double __nv_fma(double, double, double);
float __nv_fmaf(float, float, float);
double __nv_fmax(double, double);
float __nv_fmaxf(float, float);
double __nv_fmin(double, double);
float __nv_fminf(float, float);
double __nv_floor(double);
float __nv_floorf(float);
double __nv_fmod(double, double);
float __nv_fmodf(float, float);
double __nv_frexp(double, int *);
float __nv_frexpf(float, int *);
double __nv_hypot(double, double);
float __nv_hypotf(float, float);
int __nv_ilogb(double);
int __nv_ilogbf(float);
double __nv_ldexp(double, int);
float __nv_ldexpf(float, int);
long long __nv_llrint(double);
long long __nv_llrintf(float);
long long __nv_llround(double);
long long __nv_llroundf(float);
double __nv_log10(double);
float __nv_log10f(float);
double __nv_log1p(double);
float __nv_log1pf(float);
double __nv_log2(double);
float __nv_log2f(float);
double __nv_logb(double);
float __nv_logbf(float);
double __nv_log(double);
float __nv_logf(float);
long __nv_lrint(double);
long __nv_lrintf(float);
long __nv_lround(double);
long __nv_lroundf(float);
double __nv_modf(double, double *);
float __nv_modff(float, float *);
double __nv_nearbyint(double);
float __nv_nearbyintf(float);
double __nv_nextafter(double, double);
float __nv_nextafterf(float, float);
double __nv_pow(double, double);
float __nv_powf(float, float);
double __nv_remainder(double, double);
float __nv_remainderf(float, float);
double __nv_remquo(double, double, int *);
float __nv_remquof(float, float, int *);
double __nv_rint(double);
float __nv_rintf(float);
double __nv_round(double);
float __nv_roundf(float);
double __nv_scalb(double, int);
float __nv_scalbf(float, int);
double __nv_sin(double);
float __nv_sinf(float);
void __nv_sincos(double, double *, double *);
void __nv_sincosf(float, float *, float *);
double __nv_sinh(double);
float __nv_sinhf(float);
double __nv_sqrt(double);
float __nv_sqrtf(float);
double __nv_tan(double);
float __nv_tanf(float);
double __nv_tanh(double);
float __nv_tanhf(float);
double __nv_tgamma(double);
float __nv_tgammaf(float);
double __nv_trunc(double);
float __nv_truncf(float);
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_MATH_GPU_NVPTX_DECLARATIONS_H
