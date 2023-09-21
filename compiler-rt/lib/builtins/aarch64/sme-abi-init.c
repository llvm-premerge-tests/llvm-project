// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

__attribute__((visibility("hidden"), nocommon))
unsigned long __aarch64_has_sme;

__attribute__((visibility("hidden"), nocommon))
unsigned long __aarch64_has_tpidr2_el0;

// We have multiple ways to check that the function has SME, depending on our
// target.
// * For Linux we can use getauxval().
// * For newlib we can use __aarch64_sme_accessible().

#if defined(__linux__) && defined(COMPILER_RT_HAS_HWCAP2_SME)

#include <sys/auxv.h>

static unsigned long has_sme(void) {
  return getauxval(AT_HWCAP2) &  (/*HWCAP2_SME*/1 << 23);
}
#else
__attribute__((weak))
extern unsigned long __aarch64_sme_accessible(void);

static unsigned long has_sme(void)  {
  if (__aarch64_sme_accessible)
    return __aarch64_sme_accessible();
  return 0;
}
#endif

__attribute__((constructor(90)))
static void init_aarch64_has_sme(void) {
  __aarch64_has_sme = has_sme();
}
__attribute__((constructor(90)))
static void init_aarch64_has_tpidr2_el0(void) {
  __aarch64_has_tpidr2_el0 = has_sme();
}
