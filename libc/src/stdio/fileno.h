//===-- Implementation header of fileno -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_SRC_STDIO_FILENO_H
#define LLVM_LIBC_SRC_STDIO_FILENO_H

#include <stdio.h>

namespace __llvm_libc {

int fileno(::FILE *stream);

} // namespace __llvm_libc

#endif // LLVM_LIBC_SRC_STDIO_FILENO_H
