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

// std::views::enumerate

#include <cassert>
#include <ranges>
#include <string_view>

#include "types.h"

// Concepts

template <class View, class T>
concept CanBePiped = requires(View&& view, T&& t) {
  { std::forward<View>(view) | std::forward<T>(t) };
};

// Helpers

template <typename View, typename T = int>
constexpr void compareViews(View v, std::initializer_list<std::tuple<ptrdiff_t, T>> list) {
  auto b1 = v.begin();
  auto e1 = v.end();
  auto b2 = list.begin();
  auto e2 = list.end();
  for (; b1 != e1 && b2 != e2; ++b1, ++b2) {
    assert(*b1 == *b2);
  }
  assert(b1 == e1);
  assert(b2 == e2);
}

// Test SFINAE friendliness

static_assert(CanBePiped<Range, decltype(std::views::enumerate)>);

static_assert(!std::is_invocable_v<decltype(std::views::enumerate)>);
static_assert(std::is_invocable_v<decltype(std::views::enumerate), Range>);
static_assert(!std::is_invocable_v<decltype(std::views::enumerate), NotAView>);
static_assert(!std::is_invocable_v<decltype(std::views::enumerate), NotInvocable>);

static_assert(std::is_same_v<decltype(std::ranges::views::enumerate), decltype(std::views::enumerate)>);

constexpr bool test() {
  // Test `views::enumerate_view(v)`
  {
    int buff[] = {0, 1, 2, 3};

    using Result = std::ranges::enumerate_view<Range>;
    Range const range(buff, buff + 4);

    std::same_as<Result> decltype(auto) result = std::views::enumerate(range);
    compareViews(result, {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  }
  {
    std::string_view sv{"babazmt"};
    using Result = std::ranges::enumerate_view<std::string_view>;

    std::same_as<Result> decltype(auto) result = std::views::enumerate(sv);
    compareViews(result, {{0, 'b'}, {1, 'a'}, {2, 'b'}, {3, 'a'}, {4, 'z'}, {5, 'm'}, {6, 't'}});
  }
  // Test `adaptor | views::enumerate`
  {
    int buff[] = {0, 1, 2, 3};

    using Result = std::ranges::enumerate_view<Range>;
    Range const range(buff, buff + 4);

    std::same_as<Result> decltype(auto) result = range | std::views::enumerate;
    compareViews(result, {{0, 0}, {1, 1}, {2, 2}, {3, 3}});
  }
  {
    std::string_view sv{"babazmt"};
    using Result = std::ranges::enumerate_view<std::string_view>;

    std::same_as<Result> decltype(auto) result = sv | std::views::enumerate;
    compareViews(result, {{0, 'b'}, {1, 'a'}, {2, 'b'}, {3, 'a'}, {4, 'z'}, {5, 'm'}, {6, 't'}});
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
