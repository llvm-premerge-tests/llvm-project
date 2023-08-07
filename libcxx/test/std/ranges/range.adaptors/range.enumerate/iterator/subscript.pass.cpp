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

// constexpr auto operator[](difference_type n) const
//   requires random_access_range<Base>
// { return reference-type(pos_ + n, current_[n]); }

#include <cassert>
#include <ranges>
#include <tuple>

#include "test_iterators.h"

template <class T, class U>
concept HasSubscriptOperator = requires(T t, U u) { t[u]; };

template <class BaseRange>
using EnumerateIterator = std::ranges::iterator_t<std::ranges::enumerate_view<BaseRange>>;

using RandomAccessRange = std::ranges::subrange<std::tuple<int>*>;
static_assert(std::ranges::random_access_range<RandomAccessRange>);

static_assert(HasSubscriptOperator<EnumerateIterator<RandomAccessRange>, int>);

using BidirectionalRange = std::ranges::subrange<bidirectional_iterator<int*>>;
static_assert(!std::ranges::random_access_range<BidirectionalRange>);

static_assert(!HasSubscriptOperator<EnumerateIterator<BidirectionalRange>, int>);

constexpr bool test() {
  // Reference
  {
    int ts[]  = {0, 1, 2, 3, 84};
    auto view = ts | std::views::enumerate;
    auto it   = view.begin();

    assert(std::get<0>(it[0]) == std::get<0>(*it));
    assert(std::get<1>(it[0]) == std::get<1>(*it));
    assert(std::get<0>(it[2]) == std::get<0>(*(it + 2)));
    assert(std::get<1>(it[2]) == std::get<1>(*(it + 2)));
    assert(std::get<0>(it[4]) == std::get<0>(*(it + 4)));
    assert(std::get<1>(it[4]) == std::get<1>(*(it + 4)));

    static_assert(std::is_same_v<decltype(it[2]), std::tuple<std::ptrdiff_t, int&>>);
    static_assert(std::is_same_v<decltype(it[2]), std::tuple<decltype(it)::difference_type, int&>>);
  }

  // Value
  {
    auto view = std::views::iota(0, 5) | std::views::enumerate;
    auto it   = view.begin();

    assert(it[0] == *it);
    assert(it[2] == *(it + 2));
    assert(it[4] == *(it + 4));

    static_assert(std::is_same_v<decltype(it[2]), std::tuple<std::ptrdiff_t, int>>);
    static_assert(std::is_same_v<decltype(it[2]), std::tuple<decltype(it)::difference_type, int>>);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
