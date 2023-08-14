//===-- Implementation of fileno ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/stdio/fileno.h"
#include "src/__support/File/file.h"

#include "src/errno/libc_errno.h"
#include <stdio.h>

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, fileno, (::FILE *__restrict stream)) {
  auto fd = get_fileno(reinterpret_cast<__llvm_libc::File *>(stream));

  if (fd >= 0) {
    return fd;
  } else {
    errno = EBADF;
    return -1;
  }
}

} // namespace __llvm_libc
