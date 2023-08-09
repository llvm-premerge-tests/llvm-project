// RUN: %clang_cc1 %s -ffreestanding -triple=x86_64-unknown-unknown -target-feature +avx10.1 -emit-llvm -verify

#include <immintrin.h>

__m512d test_mm512_sqrt_pd(__m512d a)
{
  // CHECK-LABEL: @test_mm512_sqrt_pd
  return __builtin_ia32_sqrtpd512(a, _MM_FROUND_CUR_DIRECTION); // expected-error {{'__builtin_ia32_sqrtpd512' needs target feature avx10-512bit}}
}
