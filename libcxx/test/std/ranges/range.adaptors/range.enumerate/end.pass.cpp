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

// constexpr auto end() requires (!simple-view<V>) {
//   if constexpr (common_range<V> && sized_range<V>)
//     return iterator<false>(ranges::end(base_), ranges::distance(base_));
//   else
//     return sentinel<false>(ranges::end(base_));
// }
// constexpr auto end() const requires range-with-movable-references<const V> {
//   if constexpr (common_range<const V> && sized_range<const V>)
//     return iterator<true>(ranges::end(base_), ranges::distance(base_));
//   else
//     return sentinel<true>(ranges::end(base_));
// }

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "types.h"

constexpr bool test() {
  int buff[] = {1, 2, 3, 4, 5, 6, 7, 8};

  // Check the return type of `.end()`
  {
    Range range(buff, buff + 1);

    std::ranges::enumerate_view view(range);
    using Iterator = std::ranges::iterator_t<decltype(view)>;
    ASSERT_SAME_TYPE(Iterator, decltype(view.end()));
  }
  // begin() over an empty range
  {
    Range range(buff, buff);

    std::ranges::enumerate_view view(range);
    auto it = view.begin();
    assert(base(it.base()) == buff);
    assert(it == view.end());
  }
  // begin() over a 1-element range
  {
    Range range(buff, buff + 1);

    std::ranges::enumerate_view view(range);
    auto it = view.end();
    assert(base(it.base()) == buff + 1);
  }
  // begin() over a N-element range
  {
    Range range(buff, buff + 8);

    std::ranges::enumerate_view view(range);
    auto it = view.end();
    assert(base(it.base()) == buff + 8);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
