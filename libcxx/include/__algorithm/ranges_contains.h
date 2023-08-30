//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _LIBCPP___ALGORITHM_RANGES_CONTAINS_H
#define _LIBCPP___ALGORITHM_RANGES_CONTAINS_H

#include <__algorithm/in_in_result.h>
#include <__algorithm/ranges_starts_with.h>
#include <__config>
#include <__functional/identity.h>
#include <__functional/ranges_operations.h>
#include <__iterator/concepts.h>
#include <__iterator/indirectly_comparable.h>
#include <__ranges/access.h>
#include <__ranges/concepts.h>
#include <__utility/move.h>

#if !defined(_LIBCPP_HAS_NO_PRAGMA_SYSTEM_HEADER)
#  pragma GCC system_header
#endif

#if _LIBCPP_STD_VER >= 23

_LIBCPP_BEGIN_NAMESPACE_STD

namespace ranges {
namespace __contains {
struct __fn {
  template <class _Iter, class _Sent, class _Type, class _Proj>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr bool
  __compare_single_element(_Iter __first, _Sent __last, const _Type& __value, _Proj& __proj = {}) const {
    auto __pred = [&](auto&& __e) { return std::forward<decltype(__e)>(__e) == __value; };
    for (; __first != __last; ++__first) {
      if (std::invoke(std::move(__pred), std::invoke(std::move(__proj), *__first)))
        return true;
    }
    return false;
  }

  template <input_iterator _Iter, sentinel_for<_Iter> _Sent, class _Type, class _Proj = identity>
    requires indirect_binary_predicate<ranges::equal_to, projected<_Iter, _Proj>, const _Type*>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr bool
  operator()(_Iter __first, _Sent __last, const _Type& __value, _Proj __proj = {}) const {
    return __compare_single_element(std::move(__first), std::move(__last), __value, __proj);
  }

  template <input_range _Range, class _Type, class _Proj = identity>
    requires indirect_binary_predicate<ranges::equal_to, projected<iterator_t<_Range>, _Proj>, const _Type*>
  _LIBCPP_NODISCARD_EXT _LIBCPP_HIDE_FROM_ABI constexpr bool
  operator()(_Range&& __range, const _Type& __value, _Proj __proj = {}) const {
    return __compare_single_element(ranges::begin(__range), ranges::end(__range), __value, __proj);
  }
};
} // namespace __contains
inline namespace __cpo {
inline constexpr auto contains = __contains::__fn{};
} // namespace __cpo
} // namespace ranges

_LIBCPP_END_NAMESPACE_STD

#endif // _LIBCPP_STD_VER >= 23

#endif // _LIBCPP___ALGORITHM_RANGES_CONTAINS_H
