//===------------- AMDGPU implementation of timing utils --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_TIMING_AMDGPU
#define LLVM_LIBC_UTILS_GPU_TIMING_AMDGPU

#include "src/__support/GPU/utils.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stdint.h>

namespace __llvm_libc {

// Returns the overhead associated with calling the profiling region. This
// allows us to substract the constant-time overhead from the latency to
// obtain a true result. This can vary with system load.
[[gnu::noinline]] static LIBC_INLINE uint64_t overhead() {
  __builtin_amdgcn_s_waitcnt(0);
  uint64_t start = gpu::processor_clock();
  __builtin_amdgcn_s_waitcnt(0);
  uint32_t result = 0.0;
  asm volatile("v_or_b32 %[v_reg], 0, %[v_reg]\n" ::[v_reg] "v"(result) :);
  asm volatile("" ::"r"(start));
  uint64_t stop = gpu::processor_clock();
  __builtin_amdgcn_s_waitcnt(0);
  return stop - start;
}

// Stimulate a simple function and obtain its latency in clock cycles on the
// system. This function cannot be inlined or else it will disturb the very
// deliccate balance of hard-coded dependencies.
template <typename F, typename T>
[[gnu::noinline]] static LIBC_INLINE uint64_t latency(F f, T t) {
  // We need to store the input somewhere to guarntee that the compiler will not
  // constant propagate it and remove the profiling region.
  volatile uint32_t storage = t;
  float arg = storage;
  asm volatile("" ::"r"(arg));
  // The AMDGPU architecture needs to wait on pending results.
  __builtin_amdgcn_s_waitcnt(0);
  // Get the current timestamp from the clock.
  uint64_t start = gpu::processor_clock();
  __builtin_amdgcn_s_waitcnt(0);

  // This forces the compiler to load the input argument and run the clock cycle
  // counter before the profiling region.
  asm volatile("" ::"r"(arg), "r"(start));

  // Run the function under test and return its value.
  auto result = f(arg);

  // This inline assembly performs a no-op which forces the result to both be
  // used and prevents us from exiting this region before it's complete.
  asm volatile("v_or_b32 %[v_reg], 0, %[v_reg]\n" ::[v_reg] "v"(result) :);

  // Obtain the current timestamp after running the calculation and force
  // ordering.
  uint64_t stop = gpu::processor_clock();
  asm volatile("" ::"r"(stop));
  __builtin_amdgcn_s_waitcnt(0);

  // Return the time elapsed.
  return stop - start;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_GPU_TIMING_AMDGPU
