//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_TYPES_H

#include <ranges>

#include "test_iterators.h"

// Types

struct Range : std::ranges::view_base {
  using Iterator = cpp20_input_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;

  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Range(Range const& other) : begin_(other.begin_), end_(other.end_), wasCopyInitialized(true) {}
  constexpr Range(Range&& other) : begin_(other.begin_), end_(other.end_), wasMoveInitialized(true) {}
  Range& operator=(Range const&) = default;
  Range& operator=(Range&&)      = default;

  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

  int* begin_;
  int* end_;

  bool wasCopyInitialized = false;
  bool wasMoveInitialized = false;
};

static_assert(std::ranges::__range_with_movable_references<Range>);
static_assert(std::ranges::range<Range> && std::ranges::view<Range>);

struct SizedRange : public Range {
  std::size_t size_;

  constexpr std::size_t size() { return size_; }
};

struct JustAView : std::ranges::view_base {
  JustAView() = default;

  forward_iterator<int*> begin() const;
  sentinel_wrapper<forward_iterator<int*>> end() const;
};

static_assert(std::ranges::view<JustAView>);

template <class Iterator, class Sentinel>
struct MinimalView : std::ranges::view_base {
  constexpr explicit MinimalView(Iterator it, Sentinel sent) : it_(base(std::move(it))), sent_(base(std::move(sent))) {}

  MinimalView(MinimalView&&)            = default;
  MinimalView& operator=(MinimalView&&) = default;

  constexpr Iterator begin() const { return Iterator(it_); }
  constexpr Sentinel end() const { return Sentinel(sent_); }

private:
  decltype(base(std::declval<Iterator>())) it_;
  decltype(base(std::declval<Sentinel>())) sent_;
};

struct NotInvocable {};

struct NotAView {};

static_assert(!std::ranges::view<NotAView>);

struct NotAViewRange {
  using Iterator = cpp20_input_iterator<int*>;
  using Sentinel = sentinel_wrapper<Iterator>;

  NotAViewRange() = default;
  constexpr explicit NotAViewRange(int* b, int* e) : begin_(b), end_(e) {}
  constexpr NotAViewRange(NotAViewRange const& other) = default;
  constexpr NotAViewRange(NotAViewRange&& other)      = default;
  NotAViewRange& operator=(NotAViewRange const&)      = default;
  NotAViewRange& operator=(NotAViewRange&&)           = default;

  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

  int* begin_;
  int* end_;
};

static_assert(std::ranges::range<NotAViewRange> && !std::ranges::view<NotAViewRange>);

template <bool IsNoexcept>
class NoexceptIterMoveInputIterator {
  int* it_;

public:
  using iterator_category = std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = typename std::iterator_traits<int*>::difference_type;
  using pointer           = int*;
  using reference         = int&;

  NoexceptIterMoveInputIterator() = default;
  explicit constexpr NoexceptIterMoveInputIterator(int* it) : it_(it) {}

  friend constexpr decltype(auto) iter_move(const NoexceptIterMoveInputIterator& it) noexcept(IsNoexcept) {
    return std::ranges::iter_move(it.it_);
  }

  friend constexpr int* base(const NoexceptIterMoveInputIterator& i) { return i.it_; }

  constexpr reference operator*() const { return *it_; }
  constexpr NoexceptIterMoveInputIterator& operator++() {
    ++it_;
    return *this;
  }
  constexpr NoexceptIterMoveInputIterator operator++(int) {
    NoexceptIterMoveInputIterator tmp(*this);
    ++(*this);
    return tmp;
  }
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_ENUMERATE_TYPES_H
