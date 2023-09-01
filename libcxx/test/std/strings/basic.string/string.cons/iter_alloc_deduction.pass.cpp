//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>
// UNSUPPORTED: c++03, c++11, c++14

// template<class InputIterator,
//      class Allocator = allocator<typename iterator_traits<InputIterator>::value_type>>
//  basic_string(InputIterator, InputIterator, Allocator = Allocator())
//    -> basic_string<typename iterator_traits<InputIterator>::value_type,
//                 char_traits<typename iterator_traits<InputIterator>::value_type>,
//                 Allocator>; // constexpr since C++20
//
//  The deduction guide shall not participate in overload resolution if InputIterator
//  is a type that does not qualify as an input iterator, or if Allocator is a type
//  that does not qualify as an allocator.

#include <cassert>
#include <cstddef>
#include <iterator>
#include <string>
#include <type_traits>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

class NotAnIterator {};
using NotAnInputIterator = std::back_insert_iterator<std::basic_string<char16_t> >;

template <typename T>
struct NotAnAllocator {
  typedef T value_type;
};

template <class Iter, class Alloc, class = void>
struct CanDeduce : std::false_type {};

template <class Iter, class Alloc>
struct CanDeduce<Iter,
                 Alloc,
                 decltype((void)std::basic_string{std::declval<Iter>(), std::declval<Iter>(), std::declval<Alloc>()})>
    : std::true_type {};

static_assert(CanDeduce<char*, std::allocator<char>>::value);
static_assert(!CanDeduce<NotAnIterator, std::allocator<char>>::value);
static_assert(!CanDeduce<NotAnInputIterator, std::allocator<char16_t>>::value);
static_assert(!CanDeduce<char*, NotAnAllocator<char>>::value);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
static_assert(CanDeduce<wchar_t*, std::allocator<wchar_t>>::value);
static_assert(!CanDeduce<wchar_t const*, NotAnAllocator<wchar_t>>::value);
#endif

template <class CharT, template <class> class Alloc>
TEST_CONSTEXPR_CXX20 void
test_string(const CharT* s, const std::basic_string<CharT, std::char_traits<CharT>, Alloc<CharT> > s1) {
  using S = decltype(s1); // what type did we get?
  static_assert(std::is_same_v<typename S::value_type, CharT>, "");
  static_assert(std::is_same_v<typename S::traits_type, std::char_traits<CharT> >, "");
  static_assert(std::is_same_v<typename S::allocator_type, Alloc<CharT> >, "");
  assert(s1.size() == 10);
  assert(s1.compare(0, s1.size(), s, s1.size()) == 0);
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    const char* s = "12345678901234";
    std::basic_string s1(s, s + 10); // Can't use {} here
    test_string<char, std::allocator>(s, std::move(s1));
  }
  {
    const char* s = "12345678901234";
    std::basic_string s1{s, s + 10, std::allocator<char>{}};
    test_string<char, std::allocator>(s, std::move(s1));
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    const wchar_t* s = L"12345678901234";
    std::basic_string s1{s, s + 10, test_allocator<wchar_t>{}};
    test_string<wchar_t, test_allocator>(s, std::move(s1));
  }
  {
    const char16_t* s = u"12345678901234";
    std::basic_string s1{s, s + 10, min_allocator<char16_t>{}};
    test_string<char16_t, min_allocator>(s, std::move(s1));
  }
#endif
  {
    const char32_t* s = U"12345678901234";
    std::basic_string s1{s, s + 10, explicit_allocator<char32_t>{}};
    test_string<char32_t, explicit_allocator>(s, std::move(s1));
  }

  return true;
}

int main(int, char**) {
  test();
#if TEST_STD_VER > 17
  static_assert(test());
#endif

  return 0;
}
