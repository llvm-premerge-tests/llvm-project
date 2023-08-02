//===-- Linux implementation of wait4 -------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/common.h"
#include "src/__support/libc_assert.h"

#include "src/sys/wait/wait4.h"
#include "src/sys/wait/wait4Impl.h"

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(pid_t, wait4,
                   (pid_t pid, int *wait_status, int options,
                    struct rusage *usage)) {
  return internal::wait4impl(pid, wait_status, options, usage);
}

} // namespace __llvm_libc
