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

// friend constexpr auto iter_move(const iterator& i)
//   noexcept(noexcept(ranges::iter_move(i.current_)) &&
//             is_nothrow_move_constructible_v<range_rvalue_reference_t<Base>>) {
//   return tuple<difference_type,
//                 range_rvalue_reference_t<Base>>(i.pos_, ranges::iter_move(i.current_));
// }

#include <ranges>
#include <array>
#include <cassert>
#include <utility>

#include "test_iterators.h"
#include "test_macros.h"
#include "../types.h"

template <class Iterator, bool HasNoexceptIterMove>
constexpr void test() {
  using Sentinel          = sentinel_wrapper<Iterator>;
  using View              = MinimalView<Iterator, Sentinel>;
  using EnumerateView     = std::ranges::enumerate_view<View>;
  using EnumerateIterator = std::ranges::iterator_t<EnumerateView>;

  auto make_enumerate_view = [](auto begin, auto end) {
    View view{Iterator(begin), Sentinel(Iterator(end))};

    return EnumerateView(std::move(view));
  };

  std::array<int, 5> array{0, 1, 2, 3, 4};
  EnumerateView view         = make_enumerate_view(array.begin(), array.end());
  EnumerateIterator const it = view.begin();

  auto&& result = iter_move(it);
  assert(get<0>(result) == 0);
  assert(&get<1>(result) == array.begin());

  static_assert(noexcept(iter_move(it)) == HasNoexceptIterMove);
}

constexpr bool tests() {
  // clang-format off
  test<cpp17_input_iterator<int*>,           /* noexcept */ false>();
  test<cpp20_input_iterator<int*>,           /* noexcept */ false>();
  test<forward_iterator<int*>,               /* noexcept */ false>();
  test<bidirectional_iterator<int*>,         /* noexcept */ false>();
  test<random_access_iterator<int*>,         /* noexcept */ false>();
  test<contiguous_iterator<int*>,            /* noexcept */ false>();
  test<int*,                                 /* noexcept */ true>();
  test<NoexceptIterMoveInputIterator<true>,  /* noexcept */ true>();
  test<NoexceptIterMoveInputIterator<false>, /* noexcept */ false>();
  // clang-format on

  return true;
}

int main(int, char**) {
  tests();
  static_assert(tests());
  return 0;
}
