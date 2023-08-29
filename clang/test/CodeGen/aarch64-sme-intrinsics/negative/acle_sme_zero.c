// REQUIRES: aarch64-registered-target
// RUN: %clang_cc1 -triple aarch64-none-linux-gnu -target-feature +sme -fsyntax-only -verify %s

#include <arm_sme.h>

__attribute__((arm_streaming, arm_shared_za))
void test_svzero_mask_za(void) {
  // expected-error@+1 {{argument value 256 is outside the valid range [0, 255]}}
  svzero_mask_za(256);
}
