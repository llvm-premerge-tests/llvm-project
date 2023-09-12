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

// friend constexpr bool operator==(const iterator& x, const iterator& y) noexcept;

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "../types.h"

constexpr bool test() {
  int buff[] = {0, 1, 2, 3};
  {
    using View = std::ranges::enumerate_view<Range>;
    Range const range(buff, buff + 4);

    std::same_as<View> decltype(auto) ev = std::views::enumerate(range);

    auto it1 = ev.begin();
    auto it2 = it1 + 3;
    auto it3 = it1 + 4;

    assert(it1 == it1);
    assert(it1 != it2);
    assert(it3 == ev.end());
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
