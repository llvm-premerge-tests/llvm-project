//===------------- NVPTX implementation of timing utils ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIBC_UTILS_GPU_TIMING_NVPTX
#define LLVM_LIBC_UTILS_GPU_TIMING_NVPTX

#include "src/__support/GPU/utils.h"
#include "src/__support/common.h"
#include "src/__support/macros/attributes.h"
#include "src/__support/macros/config.h"

#include <stdint.h>

namespace __llvm_libc {

// Returns the overhead associated with calling the profiling region. This
// allows us to substract the constant-time overhead from the latency to
// obtain a true result. This can vary with system load.
[[gnu::noinline]] static uint64_t overhead() {
  volatile uint32_t x = 1;
  uint32_t y = x;
  uint64_t start = gpu::processor_clock();
  asm volatile("" ::"r"(y), "r"(start));
  uint32_t result = y;
  asm volatile("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(result) :);
  uint64_t stop = gpu::processor_clock();
  volatile auto storage = result;
  return stop - start;
}

// Stimulate a simple function and obtain its latency in clock cycles on the
// system. This function cannot be inlined or else it will disturb the very
// deliccate balance of hard-coded dependencies.
//
// FIXME: This does not work in general on NVPTX because of further
// optimizations ptxas performs. The only way to get consistent results is to
// pass and extra "SHELL:-Xcuda-ptxas -O0" to CMake's compiler flag. This
// negatively implacts performance but it is at least stable.
template <typename F, typename T>
[[gnu::noinline]] static LIBC_INLINE uint64_t latency(F f, T t) {
  // We need to store the input somewhere to guarantee that the compiler will
  // not constant propagate it and remove the profiling region.
  volatile T storage = t;
  T arg = storage;
  asm volatile("" ::"r"(arg));

  // Get the current timestamp from the clock.
  uint64_t start = gpu::processor_clock();

  // This forces the compiler to load the input argument and run the clock cycle
  // counter before the profiling region.
  asm volatile("" ::"r"(arg), "r"(start));

  // Run the function under test and return its value.
  auto result = f(arg);

  // This inline assembly performs a no-op which forces the result to both be
  // used and prevents us from exiting this region before it's complete.
  asm volatile("or.b32 %[v_reg], %[v_reg], 0;" ::[v_reg] "r"(result) :);

  // Obtain the current timestamp after running the calculation and force
  // ordering.
  uint64_t stop = gpu::processor_clock();
  asm volatile("" ::"r"(stop));
  volatile T output = result;

  // Return the time elapsed.
  return stop - start;
}

} // namespace __llvm_libc

#endif // LLVM_LIBC_UTILS_GPU_TIMING_NVPTX
