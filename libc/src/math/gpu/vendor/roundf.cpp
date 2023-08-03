//===-- Implementation of the GPU roundf function -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/math/roundf.h"
#include "src/__support/common.h"

#include "common.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(float, roundf, (float x)) { return internal::roundf(x); }

} // namespace __llvm_libc
