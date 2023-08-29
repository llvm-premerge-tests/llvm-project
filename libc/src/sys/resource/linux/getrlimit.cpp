//===-- Linux implementation of getrlimit ---------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/sys/resource/getrlimit.h"

#include "src/__support/OSUtil/syscall.h" // For internal syscall function.
#include "src/__support/common.h"

#include "src/errno/libc_errno.h"
#include <sys/resource.h> // For struct rlimit
#include <sys/syscall.h>  // For syscall numbers.

namespace __llvm_libc {

LLVM_LIBC_FUNCTION(int, getrlimit, (int res, struct rlimit *limits)) {
#if defined(LIBC_TARGET_ARCH_IS_RISCV32) || defined(LIBC_TARGET_ARCH_IS_ARM32)
  // SYS_prlimit64 takes a rlimit64 struct, which is a struct rlimit with
  // 64-bit members. In 32-bit systems struct rlimit has 32-bit members, so
  // we need to fill a rlimit64 struct with the 32-bit values from struct rlimit
  struct rlimit64 lim;
#else
  struct rlimit lim;
#endif
  lim.rlim_cur = limits->rlim_cur;
  lim.rlim_max = limits->rlim_max;

  int ret =
      __llvm_libc::syscall_impl<int>(SYS_prlimit64, 0, res, nullptr, &lim);
  if (ret < 0) {
    libc_errno = -ret;
    return -1;
  }

  limits->rlim_cur = static_cast<rlim_t>(lim.rlim_cur);
  limits->rlim_max = static_cast<rlim_t>(lim.rlim_max);
  return 0;
}

} // namespace __llvm_libc
