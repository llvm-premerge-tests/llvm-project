//===-- Definition of the global stderr object ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/macros/attributes.h"

#include <stdio.h>

#ifdef LIBC_TARGET_ARCH_IS_GPU
static struct {
} stub;
namespace __llvm_libc {
FILE *stderr = reinterpret_cast<FILE *>(&stub);
}
extern "C" FILE *stderr = reinterpret_cast<FILE *>(&stub);
#endif
