//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <ranges>

// class enumerate_view

// constexpr V base() const & requires copy_constructible<V> { return base_; }
// constexpr V base() && { return std::move(base_); }

#include <cassert>
#include <ranges>

#include "types.h"

constexpr bool test() {
  // Check the const& overload
  {
    int buff[] = {0, 1, 2, 3};

    using Result = std::ranges::enumerate_view<Range>;
    Range const range(buff, buff + 4);

    std::ranges::enumerate_view<Range> const view{range};
    std::same_as<Range> decltype(auto) result = view.base();
    assert(result.wasCopyInitialized);
    assert(range.begin() == result.begin());
    assert(range.end() == result.end());
  }
  // Check the && overload
  {
    int buff[] = {0, 1, 2, 3};

    using Result = std::ranges::enumerate_view<Range>;
    Range const range(buff, buff + 4);

    std::ranges::enumerate_view<Range> view{range};
    std::same_as<Range> decltype(auto) result = std::move(view).base();
    assert(result.wasMoveInitialized);
    assert(range.begin() == result.begin());
    assert(range.end() == result.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
