// RUN: %clang_cc1  -triple aarch64-none-linux-gnu -target-feature +sve -target-feature +sme -fsyntax-only -verify %s

// REQUIRES: aarch64-registered-target

#include "arm_sme.h"

void requires_za_state(void) {
  // expected-warning@+1 {{builtin call is not valid when calling from a function without active ZA state}}
  return svzero_mask_za(0);
}

// No diagnostic expected
__attribute__((arm_new_za)) void with_new_za_state(void) {
  return svzero_mask_za(0);
}

// No diagnostic expected
__attribute__((arm_shared_za)) void with_shared_za_state(void) {
  return svzero_mask_za(0);
}

// FIXME: svzero_za() is defined in arm_sme.h, but is not a builtin.
// Clang should give a similar diagnostic for 'regular' calls to shared_za
// functions when the caller doesn't have ZA state.
__attribute__((arm_shared_za)) void requires_za_state_svzero_no_mask(void) {
  return svzero_za();
}
