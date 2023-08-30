//===-- GPU implementation of the nanosleep function ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/time/nanosleep.h"

#include "time_utils.h"

namespace __llvm_libc {

constexpr uint64_t TICKS_PER_NS = 1000000000UL;

LLVM_LIBC_FUNCTION(int, nanosleep,
                   (const struct timespec *req, struct timespec *rem)) {
  if (!GPU_CLOCKS_PER_SEC || !req)
    return -1;

  uint64_t nsecs = req->tv_nsec + req->tv_sec * TICKS_PER_NS;

  uint64_t start = gpu::fixed_frequency_clock();
#if defined(LIBC_TARGET_ARCH_IS_NVPTX)
  LIBC_INLINE_ASM("nanosleep.u32 %0;" ::"r"(nsecs));
#elif defined(LIBC_TARGET_ARCH_IS_AMDGPU)
  uint64_t end = start + nsecs / (TICKS_PER_NS / GPU_CLOCKS_PER_SEC);
  uint64_t cur = gpu::fixed_frequency_clock();
  // The AMDGPU architecture does not provide a sleep implementation with a
  // known delay so we simply repeatedly sleep with a large value of ~960 clock
  // cycles and check until we've passed the time using the known frequency.
  __builtin_amdgcn_s_sleep(2);
  while (cur < end) {
    __builtin_amdgcn_s_sleep(15);
    cur = gpu::fixed_frequency_clock();
  }
#endif
  uint64_t stop = gpu::fixed_frequency_clock();

  // Check to make sure we slept for at least the desired duration and set the
  // remaining time if not.
  uint64_t elapsed = (stop - start) * (TICKS_PER_NS / GPU_CLOCKS_PER_SEC);
  if (elapsed < nsecs) {
    if (rem) {
      rem->tv_sec = (nsecs - elapsed) / TICKS_PER_NS;
      rem->tv_nsec = (nsecs - elapsed) % TICKS_PER_NS;
    }
    return -1;
  }

  return 0;
}

} // namespace __llvm_libc
