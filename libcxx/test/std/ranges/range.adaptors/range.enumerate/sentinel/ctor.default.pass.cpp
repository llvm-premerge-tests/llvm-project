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

// iterator() requires default_initializable<iterator_t<Base>> = default;

#include <cassert>
#include <ranges>

#include "test_iterators.h"
#include "../types.h"

template <class Iterator, class Sentinel = sentinel_wrapper<Iterator>>
constexpr void test() {
  using View              = MinimalView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateSentinel = std::ranges::sentinel_t<EnumerateView>;

  EnumerateSentinel s1;
  EnumerateSentinel s2{};

  assert(base(base(s1.base())) == base(base(s2.base())));

  static_assert(noexcept(EnumerateSentinel()));
}

constexpr bool tests() {
  test<cpp17_input_iterator<int*>>();
  test<cpp20_input_iterator<int*>>();
  test<forward_iterator<int*>>();
  test<bidirectional_iterator<int*>>();
  test<random_access_iterator<int*>>();
  test<contiguous_iterator<int*>>();
  test<int*>();

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());

  return 0;
}
