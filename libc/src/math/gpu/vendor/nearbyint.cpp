//===-- Implementation of the GPU nearbyint function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/nearbyint.h"
#include "src/__support/common.h"

#include "common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(double, nearbyint, (double x)) {
  return internal::nearbyint(x);
}

} // namespace __llvm_libc
