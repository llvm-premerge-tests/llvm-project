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

// constexpr auto begin() requires (!simple-view<V>);
// constexpr auto begin() const requires range-with-movable-references<const V>;

#include <cassert>
#include <concepts>
#include <ranges>

#include "test_iterators.h"
#include "types.h"

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the return type of begin()
  {
    RangeView range(buff, buff + 1);

    std::ranges::enumerate_view view(range);
    using Iterator = std::ranges::iterator_t<decltype(view)>;
    static_assert(std::same_as<Iterator, decltype(view.begin())>);
  }

  // begin() over an empty range
  {
    RangeView range(buff, buff);

    std::ranges::enumerate_view view(range);
    auto it = view.begin();
    assert(base(it.base()) == buff);
    assert(it == view.end());
  }

  // begin() over an 1-element range
  {
    RangeView range(buff, buff + 1);

    std::ranges::enumerate_view view(range);
    auto it = view.begin();
    assert(base(it.base()) == buff);
  }

  // begin() over an N-element range
  {
    RangeView range(buff, buff + 8);

    std::ranges::enumerate_view view(range);
    auto it = view.begin();
    assert(base(it.base()) == buff);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
