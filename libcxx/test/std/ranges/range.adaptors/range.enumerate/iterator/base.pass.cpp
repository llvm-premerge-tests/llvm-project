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

// constexpr sentinel_t<Base> base() const;

#include <ranges>

#include <array>
#include <cassert>
#include <concepts>
#include <utility>

#include "test_iterators.h"
#include "../types.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View              = MinimalView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateSentinel = std::ranges::sentinel_t<EnumerateView>;
  using EnumerateIterator = std::ranges::iterator_t<EnumerateView>;

  auto make_enumerate_view = [](auto begin, auto end) {
    View view{Iterator(begin), Sentinel(Iterator(end))};

    return EnumerateView(std::move(view));
  };

  std::array<int, 5> array{0, 1, 2, 3, 84};
  auto view = make_enumerate_view(array.begin(), array.end());

  // Test the const& version
  {
    EnumerateIterator const iter(view);
    Iterator const& result = iter.base();
    ASSERT_SAME_TYPE(Iterator const&, decltype(iter.base()));
    ASSERT_NOEXCEPT(iter.base());
    assert(base(result) == array.begin());
  }

  // Test the && version
  {
    EnumerateIterator iter(view);
    Iterator result = std::move(iter).base();
    ASSERT_SAME_TYPE(Iterator, decltype(std::move(iter).base()));
    assert(base(result) == array.begin());
  }
}

constexpr bool tests() {
  // test<cpp17_input_iterator<int*>>();
  // test<cpp20_input_iterator<int*>>();
  // test<forward_iterator<int*>>();
  // test<bidirectional_iterator<int*>>();
  // test<random_access_iterator<int*>>();
  // test<contiguous_iterator<int*>>();
  // test<int*>();
  // test<int const*>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
