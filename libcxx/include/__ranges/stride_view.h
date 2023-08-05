// -*- C++ -*-
//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___RANGES_STRIDE_VIEW_H
#define _LIBCPP___RANGES_STRIDE_VIEW_H

#include <__config>
#include <__iterator/concepts.h>
#include <__ranges/concepts.h>
#include <__ranges/div_ceil.h>
#include <__ranges/enable_borrowed_range.h>
#include <__type_traits/maybe_const.h>
#include <__utility/forward.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

_LIBCPP_BEGIN_NAMESPACE_STD

#if _LIBCPP_STD_VER >= 23

namespace ranges {

// [range.stride.view]

template <input_range _View>
  requires view<_View>
class stride_view : public view_interface<stride_view<_View>> {
  _View __base_;                       // exposition only
  range_difference_t<_View> __stride_; // exposition only
  // [range.stride.iterator], class template stride_view​::​iterator
  template <bool>
  class __iterator; // exposition only

public:
  _LIBCPP_HIDE_FROM_ABI constexpr explicit stride_view(_View __base, range_difference_t<_View> __stride)
      : __base_(std::move(__base)), __stride_(std::move(__stride)) {
    _LIBCPP_ASSERT(__stride > 0, "The stride value must be a positive number.");
  }

  _LIBCPP_HIDE_FROM_ABI constexpr _View base() const&
    requires copy_constructible<_View>
  {
    return __base_;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr _View base() && { return std::move(__base_); }

  _LIBCPP_HIDE_FROM_ABI constexpr range_difference_t<_View> stride() const noexcept { return __stride_; }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin()
    requires(!__simple_view<_View>)
  {
    return __iterator<false>(this, ranges::begin(__base_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto begin() const
    requires range<const _View>
  {
    return __iterator<true>(this, ranges::begin(__base_));
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end()
    requires(!__simple_view<_View>)
  {
    if constexpr (common_range<_View> && sized_range<_View> && forward_range<_View>) {
      auto missing = (__stride_ - ranges::distance(__base_) % __stride_) % __stride_;
      return __iterator<false>(this, ranges::end(__base_), missing);
    } else if constexpr (common_range<_View> && !bidirectional_range<_View>) {
      return __iterator<false>(this, ranges::end(__base_));
    } else {
      return default_sentinel;
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto end() const
    requires range<const _View>
  {
    if constexpr (common_range<const _View> && sized_range<const _View> && forward_range<const _View>) {
      auto missing = (__stride_ - ranges::distance(__base_) % __stride_) % __stride_;
      return __iterator<true>(this, ranges::end(__base_), missing);
    } else if constexpr (common_range<const _View> && !bidirectional_range<const _View>) {
      return __iterator<true>(this, ranges::end(__base_));
    } else {
      return default_sentinel;
    }
  }

  _LIBCPP_HIDE_FROM_ABI constexpr auto size()
    requires sized_range<_View>
  {
    return __to_unsigned_like(__div_ceil(ranges::distance(__base_), __stride_));
  }
  _LIBCPP_HIDE_FROM_ABI constexpr auto size() const
    requires sized_range<const _View>
  {
    return __to_unsigned_like(__div_ceil(ranges::distance(__base_), __stride_));
  }
};

template <class _Range>
stride_view(_Range&&, range_difference_t<_Range>) -> stride_view<views::all_t<_Range>>;

// [range.stride.iterator]

template <class>
struct __stride_view_iterator_category_base {};

template <forward_range _Base>
struct __stride_view_iterator_category_base<_Base> {
  static consteval auto __get_iterator_category() {
    using _Cat = typename iterator_traits<iterator_t<_Base>>::iterator_category;

    if constexpr (derived_from<_Cat, random_access_iterator_tag>) {
      return random_access_iterator_tag{};
    } else {
      return _Cat{};
    }
  }

  using iterator_category = decltype(__get_iterator_category()); // not always present
};

template <input_range _View>
  requires view<_View>
template <bool Const>
class stride_view<_View>::__iterator : __stride_view_iterator_category_base<_View> {
  using _Parent = __maybe_const<Const, stride_view>; // exposition only
  using _Base   = __maybe_const<Const, _View>;       // exposition only

  iterator_t<_Base> __current_         = iterator_t<_Base>(); // exposition only
  sentinel_t<_Base> __end_             = sentinel_t<_Base>(); // exposition only
  range_difference_t<_Base> __stride_  = 0;                   // exposition only
  range_difference_t<_Base> __missing_ = 0;                   // exposition only

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(
      _Parent* __parent, iterator_t<_Base> __current, range_difference_t<_Base> __missing = 0)
      : __current_(std::move(__current)),
        __end_(ranges::end(__parent->__base_)),
        __stride_(__parent->__stride_),
        __missing_(__missing) {} // exposition only

  static consteval auto __get_iterator_concept() {
    if constexpr (random_access_range<_Base>) {
      return random_access_iterator_tag{};
    } else if constexpr (bidirectional_range<_Base>) {
      return bidirectional_iterator_tag{};
    } else if constexpr (forward_range<_Base>) {
      return forward_iterator_tag{};
    } else {
      return input_iterator_tag{};
    }
  }

  friend class stride_view<_View>;

public:
  using difference_type  = range_difference_t<_Base>;
  using value_type       = range_value_t<_Base>;
  using iterator_concept = decltype(__get_iterator_concept());
  // using iterator_category - not always present

  _LIBCPP_HIDE_FROM_ABI __iterator()
    requires default_initializable<iterator_t<_Base>>
  = default;

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator(__iterator<!Const> __other)
    requires Const && convertible_to<iterator_t<_View>, iterator_t<_Base>> &&
                 convertible_to<sentinel_t<_View>, sentinel_t<_Base>>
      : __current_(std::move(__other.__current_)),
        __end_(std::move(__other.__end_)),
        __stride_(__other.__stride_),
        __missing_(__other.__missing_) {}

  _LIBCPP_HIDE_FROM_ABI constexpr iterator_t<_Base> base() && { return std::move(__current_); }
  _LIBCPP_HIDE_FROM_ABI constexpr const iterator_t<_Base>& base() const& noexcept { return __current_; }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator*() const { return *__current_; }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator++() {
    _LIBCPP_ASSERT(__current_ != __end_, "Out of bound iteration.");
    __missing_ = ranges::advance(__current_, __stride_, __end_);
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr void operator++(int) { return ++*this; }
  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator++(int)
    requires forward_range<_Base>
  {
    auto __temp = *this;
    ++*this;
    return __temp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator--()
    requires bidirectional_range<_Base>
  {
    ranges::advance(__current_, __missing_ - __stride_);
    __missing_ = 0;
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr __iterator operator--(int)
    requires bidirectional_range<_Base>
  {
    auto __temp = *this;
    --*this;
    return __temp;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator+=(difference_type __n)
    requires random_access_range<_Base>
  {
    if (__n > 0) {
      _LIBCPP_ASSERT(ranges::distance(__current_, __end_) > __stride_ * (__n - 1), "Out of bound iteration.");
      ranges::advance(__current_, __stride_ * (__n - 1));
      __missing_ = ranges::advance(__current_, __stride_, __end_);
    } else if (__n < 0) {
      ranges::advance(__current_, __stride_ * __n + __missing_);
      __missing_ = 0;
    }
    return *this;
  }
  _LIBCPP_HIDE_FROM_ABI constexpr __iterator& operator-=(difference_type __n)
    requires random_access_range<_Base>
  {
    return *this += -__n;
  }

  _LIBCPP_HIDE_FROM_ABI constexpr decltype(auto) operator[](difference_type __n) const
    requires random_access_range<_Base>
  {
    return *(*this + __n);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __i, default_sentinel_t) {
    return __i.__current_ == __i.__end;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator==(const __iterator& __x, const __iterator& __y)
    requires equality_comparable<iterator_t<_Base>>
  {
    return __x.__current_ == __y.__current_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __x.__current_ < __y._current_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return __y.__current_ < __x.__current_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator<=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return !(__y < __x);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr bool operator>=(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base>
  {
    return !(__x < __y);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr auto operator<=>(const __iterator& __x, const __iterator& __y)
    requires random_access_range<_Base> && three_way_comparable<iterator_t<_Base>>
  {
    return __x.__current_ <=> __y.__current_;
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(const __iterator& __i, difference_type __n)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r += __n;
    return __r;
  }
  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator+(difference_type __n, const __iterator& __i)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r += __n;
    return __r;
  }
  _LIBCPP_HIDE_FROM_ABI friend constexpr __iterator operator-(const __iterator& __i, difference_type __n)
    requires random_access_range<_Base>
  {
    auto __r = __i;
    __r -= __n;
    return __r;
  }
  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const __iterator& __x, const __iterator& __y)
    requires sized_sentinel_for<iterator_t<_Base>, iterator_t<_Base>>
  {
    constexpr auto __n = (__x.__current_ - __y.__current_);
    if constexpr (forward_range<_Base>) {
      return (__n + __x.__missing_ - __y.__missing_) / __x.__stride_;
    } else if constexpr (__n < 0) {
      return __div_ceil(-__n, __x.__stride_);
    } else {
      return __div_ceil(__n, __x.__stride_);
    }
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(default_sentinel_t __y, const __iterator& __x)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return __div_ceil(__x.__end_ - __x.__current_, __x.__stride_);
  }
  _LIBCPP_HIDE_FROM_ABI friend constexpr difference_type operator-(const __iterator& __x, default_sentinel_t __y)
    requires sized_sentinel_for<sentinel_t<_Base>, iterator_t<_Base>>
  {
    return -(__y - __x);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr range_rvalue_reference_t<_Base>
  iter_move(const __iterator& __i) noexcept(noexcept(ranges::iter_move(__i.__current_))) {
    return ranges::iter_move(__i.__current_);
  }

  _LIBCPP_HIDE_FROM_ABI friend constexpr void
  iter_swap(const __iterator& __x,
            const __iterator& __y) noexcept(noexcept(ranges::iter_swap(__x.__current_, __y.__current_)))
    requires indirectly_swappable<iterator_t<_Base>>
  {
    return ranges::iter_swap(__x.__current_, __y.__current_);
  }
};

template <class _View>
_LIBCPP_HIDE_FROM_ABI constexpr bool enable_borrowed_range<stride_view<_View>> = enable_borrowed_range<_View>;

namespace views {
namespace __stride {

struct __fn : __range_adaptor_closure<__fn> {
  template <class _Range>
  [[nodiscard]] _LIBCPP_HIDE_FROM_ABI constexpr auto
  operator()(_Range&& __range, range_difference_t<_Range> __stride) const
      noexcept(noexcept(stride_view<all_t<_Range&&>>(std::forward<_Range>(__range), __stride)))
          -> decltype(stride_view<all_t<_Range&&>>(std::forward<_Range>(__range), __stride)) {
    return stride_view<all_t<_Range&&>>(std::forward<_Range>(__range), __stride);
  }
};

} // namespace __stride

inline namespace __cpo {

inline constexpr auto stride = __stride::__fn{};

} // namespace __cpo
} // namespace views
} // namespace ranges

#endif // _LIBCPP_STD_VER >= 23

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP___RANGES_STRIDE_VIEW_H
