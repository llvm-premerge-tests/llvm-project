//===-- Definition of type struct rlimit64 --------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef __LLVM_LIBC_TYPES_STRUCT_RLIMIT64_H__
#define __LLVM_LIBC_TYPES_STRUCT_RLIMIT64_H__

#include <llvm-libc-types/rlim64_t.h>

struct rlimit64 {
  rlim64_t rlim_cur;
  rlim64_t rlim_max;
};

#endif // __LLVM_LIBC_TYPES_STRUCT_RLIMIT64_H__
