//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <string>
// UNSUPPORTED: c++03, c++11, c++14

// template<class charT,
//          class traits,
//          class Allocator = allocator<charT>
//          >
// basic_string(basic_string_view<charT, traits>, const Allocator& = Allocator())
//   -> basic_string<charT, traits, Allocator>;
//
//  The deduction guide shall not participate in overload resolution if Allocator
//  is a type that does not qualify as an allocator.

#include <string>
#include <string_view>
#include <iterator>
#include <memory>
#include <type_traits>
#include <cassert>
#include <cstddef>

#include "test_macros.h"
#include "test_allocator.h"
#include "min_allocator.h"

template <class StringView, class Allocator, class = void>
struct CanDeduce : std::false_type {};

template <class StringView, class Allocator>
struct CanDeduce<StringView, Allocator, decltype((void)
  std::basic_string{std::declval<StringView>(), std::declval<Allocator>()}
)> : std::true_type { };

struct NotAnAllocator {};
static_assert(CanDeduce<std::string_view, std::allocator<char> >::value);
static_assert(!CanDeduce<std::string_view, NotAnAllocator>::value);

template <class CharT, template <class> class Alloc>
TEST_CONSTEXPR_CXX20 void test(const std::basic_string_view<CharT, std::char_traits<CharT> > sv,
                               const std::basic_string<CharT, std::char_traits<CharT>, Alloc<CharT> > s1) {
  using S = decltype(s1); // what type did we get?
  static_assert(std::is_same_v<typename S::value_type, CharT>, "");
  static_assert(std::is_same_v<typename S::traits_type, std::char_traits<CharT> >, "");
  static_assert(std::is_same_v<typename S::allocator_type, Alloc<CharT> >, "");
  assert(s1.size() == sv.size());
  assert(s1.compare(0, s1.size(), sv.data(), s1.size()) == 0);
}

TEST_CONSTEXPR_CXX20 bool test() {
  {
    std::string_view sv = "12345678901234";
    std::basic_string s1(sv);
    test<char, std::allocator>(std::move(sv), std::move(s1));
  }
  {
    std::string_view sv = "12345678901234";
    std::basic_string s1{sv, std::allocator<char>{}};
    test<char, std::allocator>(std::move(sv), std::move(s1));
  }
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  {
    std::wstring_view sv = L"12345678901234";
    std::basic_string s1{sv, test_allocator<wchar_t>{}};
    test<wchar_t, test_allocator>(std::move(sv), std::move(s1));
  }
#endif
#ifndef TEST_HAS_NO_CHAR8_T
  {
    std::u8string_view sv = u8"12345678901234";
    std::basic_string s1{sv, min_allocator<char8_t>{}};
    test<char8_t, min_allocator>(std::move(sv), std::move(s1));
  }
#endif
  {
    std::u16string_view sv = u"12345678901234";
    std::basic_string s1{sv, min_allocator<char16_t>{}};
    test<char16_t, min_allocator>(std::move(sv), std::move(s1));
  }
  {
    std::u32string_view sv = U"12345678901234";
    std::basic_string s1{sv, explicit_allocator<char32_t>{}};
    test<char32_t, explicit_allocator>(std::move(sv), std::move(s1));
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
