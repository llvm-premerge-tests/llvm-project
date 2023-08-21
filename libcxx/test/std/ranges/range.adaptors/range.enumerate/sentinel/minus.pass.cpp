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

// template<bool OtherConst>
//   requires sized_sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr range_difference_t<maybe-const<OtherConst, V>>
//   operator-(const iterator<OtherConst>& x, const sentinel& y);

// template<bool OtherConst>
//   requires sized_sentinel_for<sentinel_t<Base>, iterator_t<maybe-const<OtherConst, V>>>
// friend constexpr range_difference_t<maybe-const<OtherConst, V>>
//   operator-(const sentinel& x, const iterator<OtherConst>& y);

#include <cassert>
#include <ranges>

#include "test_iterators.h"

template <bool Const>
struct Iter {
  int* it_;

  using value_type       = int;
  using difference_type  = std::ptrdiff_t;
  using iterator_concept = std::input_iterator_tag;

  constexpr decltype(auto) operator*() const { return *it_; }
  constexpr Iter& operator++() {
    ++it_;
    return *this;
  }
  constexpr void operator++(int) { ++it_; }
};

template <bool Const>
struct Sent {
  int* end_;

  constexpr bool operator==(const Iter<Const>& i) const { return i.it_ == end_; }
};

template <bool Const>
struct SizedSent {
  int* end_;

  constexpr bool operator==(const Iter<Const>& i) const { return i.it_ == end_; }

  friend constexpr auto operator-(const SizedSent& st, const Iter<Const>& it) { return st.end_ - it.it_; }

  friend constexpr auto operator-(const Iter<Const>& it, const SizedSent& st) { return it.it_ - st.end_; }
};

template <bool Const>
struct CrossSizedSent {
  int* end_;

  template <bool C>
  constexpr bool operator==(const Iter<C>& i) const {
    return i.it_ == end_;
  }

  template <bool C>
  friend constexpr auto operator-(const CrossSizedSent& st, const Iter<C>& it) {
    return st.end_ - it.it_;
  }

  template <bool C>
  friend constexpr auto operator-(const Iter<C>& it, const CrossSizedSent& st) {
    return it.it_ - st.end_;
  }
};

template <class T>
struct BufferView : std::ranges::view_base {
  T* buffer_;
  std::size_t size_;

  template <std::size_t N>
  constexpr BufferView(T (&b)[N]) : buffer_(b), size_(N) {}

  template <std::size_t N>
  constexpr BufferView(std::array<T, N>& arr) : buffer_(arr.data()), size_(N) {}
};

template <template <bool> class It, template <bool> class St>
struct Range : BufferView<int> {
  using BufferView<int>::BufferView;

  using iterator       = It<false>;
  using sentinel       = St<false>;
  using const_iterator = It<true>;
  using const_sentinel = St<true>;

  constexpr iterator begin() { return {buffer_}; }
  constexpr const_iterator begin() const { return {buffer_}; }
  constexpr sentinel end() { return sentinel{buffer_ + size_}; }
  constexpr const_sentinel end() const { return const_sentinel{buffer_ + size_}; }
};

template <class T, class U>
concept HasMinus = requires(const T t, const U u) { t - u; };

template <class BaseRange>
using EnumerateView = std::ranges::enumerate_view<BaseRange>;

template <class BaseRange>
using EnumerateIter = std::ranges::iterator_t<EnumerateView<BaseRange>>;

template <class BaseRange>
using EnumerateConstIter = std::ranges::iterator_t<const EnumerateView<BaseRange>>;

template <class BaseRange>
using EnumerateSentinel = std::ranges::sentinel_t<EnumerateView<BaseRange>>;

template <class BaseRange>
using EnumerateConstSentinel = std::ranges::sentinel_t<const EnumerateView<BaseRange>>;

constexpr void testConstraints() {
  // Base is not sized
  {
    using Base = Range<Iter, Sent>;
    static_assert(!HasMinus<EnumerateSentinel<Base>, EnumerateIter<Base>>);
    static_assert(!HasMinus<EnumerateIter<Base>, EnumerateSentinel<Base>>);

    static_assert(!HasMinus<EnumerateSentinel<Base>, EnumerateConstIter<Base>>);
    static_assert(!HasMinus<EnumerateConstIter<Base>, EnumerateSentinel<Base>>);

    static_assert(!HasMinus<EnumerateConstSentinel<Base>, EnumerateConstIter<Base>>);
    static_assert(!HasMinus<EnumerateConstIter<Base>, EnumerateConstSentinel<Base>>);

    static_assert(!HasMinus<EnumerateConstSentinel<Base>, EnumerateIter<Base>>);
    static_assert(!HasMinus<EnumerateIter<Base>, EnumerateConstSentinel<Base>>);
  }

  // Base is sized but not cross const
  {
    using Base = Range<Iter, SizedSent>;
    static_assert(HasMinus<EnumerateSentinel<Base>, EnumerateIter<Base>>);
    static_assert(HasMinus<EnumerateIter<Base>, EnumerateSentinel<Base>>);

    static_assert(!HasMinus<EnumerateSentinel<Base>, EnumerateConstIter<Base>>);
    static_assert(!HasMinus<EnumerateConstIter<Base>, EnumerateSentinel<Base>>);

    static_assert(HasMinus<EnumerateConstSentinel<Base>, EnumerateConstIter<Base>>);
    static_assert(HasMinus<EnumerateConstIter<Base>, EnumerateConstSentinel<Base>>);

    static_assert(!HasMinus<EnumerateConstSentinel<Base>, EnumerateIter<Base>>);
    static_assert(!HasMinus<EnumerateIter<Base>, EnumerateConstSentinel<Base>>);
  }

  // Base is cross const sized
  {
    using Base = Range<Iter, CrossSizedSent>;
    static_assert(HasMinus<EnumerateSentinel<Base>, EnumerateIter<Base>>);
    static_assert(HasMinus<EnumerateIter<Base>, EnumerateSentinel<Base>>);

    static_assert(HasMinus<EnumerateSentinel<Base>, EnumerateConstIter<Base>>);
    static_assert(HasMinus<EnumerateConstIter<Base>, EnumerateSentinel<Base>>);

    static_assert(HasMinus<EnumerateConstSentinel<Base>, EnumerateConstIter<Base>>);
    static_assert(HasMinus<EnumerateConstIter<Base>, EnumerateConstSentinel<Base>>);

    static_assert(HasMinus<EnumerateConstSentinel<Base>, EnumerateIter<Base>>);
    static_assert(HasMinus<EnumerateIter<Base>, EnumerateConstSentinel<Base>>);
  }
}

constexpr bool test() {
  int buffer[] = {1, 2, 3, 4, 5};

  // Base is sized but not cross const
  {
    using Base = Range<Iter, SizedSent>;
    Base base{buffer};
    auto ev         = base | std::views::enumerate;
    auto iter       = ev.begin();
    auto const_iter = std::as_const(ev).begin();
    auto sent       = ev.end();
    auto const_sent = std::as_const(ev).end();

    assert(iter - sent == -5);
    assert(sent - iter == 5);
    assert(const_iter - const_sent == -5);
    assert(const_sent - const_iter == 5);
  }

  // Base is cross const sized
  {
    using Base = Range<Iter, CrossSizedSent>;
    Base base{buffer};
    auto ev         = base | std::views::enumerate;
    auto iter       = ev.begin();
    auto const_iter = std::as_const(ev).begin();
    auto sent       = ev.end();
    auto const_sent = std::as_const(ev).end();

    assert(iter - sent == -5);
    assert(sent - iter == 5);
    assert(iter - const_sent == -5);
    assert(const_sent - iter == 5);
    assert(const_iter - sent == -5);
    assert(sent - const_iter == 5);
    assert(const_iter - const_sent == -5);
    assert(const_sent - const_iter == 5);
  }

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  return 0;
}
