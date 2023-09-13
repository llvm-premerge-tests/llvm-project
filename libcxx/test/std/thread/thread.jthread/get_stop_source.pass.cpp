//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14, c++17
// XFAIL: availability-synchronization_library-missing

// [[nodiscard]] stop_source get_stop_source() noexcept;

#include <cassert>
#include <concepts>
#include <stop_token>
#include <thread>
#include <type_traits>

#include "test_macros.h"

template <class T>
concept IsGetStopSourceNoexcept = requires(T& a) {
  { a.get_stop_source() } noexcept;
};

static_assert(IsGetStopSourceNoexcept<std::jthread>);

int main(int, char**) {
  // Represents a thread
  {
    std::jthread jt{[] {}};
    std::same_as<std::stop_source> auto result = jt.get_stop_source();
    assert(result.stop_possible());
  }

  // Does not represents a thread
  {
    std::jthread jt{};
    std::same_as<std::stop_source> auto result = jt.get_stop_source();
    assert(!result.stop_possible());
  }

  return 0;
}
