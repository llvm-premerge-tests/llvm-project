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

// [[nodiscard]] bool joinable() const noexcept;

#include <cassert>
#include <concepts>
#include <thread>
#include <type_traits>

#include "test_macros.h"

template <class T>
concept IsJoinableNoexcept = requires(const T& a) {
  { a.joinable() } noexcept;
};

static_assert(IsJoinableNoexcept<std::jthread>);

int main(int, char**) {

  // Default constructed
  {
    const std::jthread jt;
    std::same_as<bool> auto result = jt.joinable();
    assert(!result);
  }

  // Non-default constructed
  {
    const std::jthread jt{[]{}};
    std::same_as<bool> auto result = jt.joinable();
    assert(result);
  }

  return 0;
}
