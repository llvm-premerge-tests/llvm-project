//===-- Implementation header of fdopen --------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FDOPEN_H
#define LLVM_LIBC_SRC_STDIO_FDOPEN_H

#include <stdio.h>

namespace __llvm_libc {

::FILE *fdopen(int fd, const char *__restrict mode);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_FDOPEN_H
