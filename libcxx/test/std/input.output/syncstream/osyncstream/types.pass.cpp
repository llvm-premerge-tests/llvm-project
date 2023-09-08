//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14 ,c++17

// <iosfwd>
// <syncstream>

#include <iosfwd>
#include <syncstream>
#include <cassert>

#include "test_macros.h"

template <class CharT>
void test_member_types() {
  using Buf = std::basic_syncbuf<CharT>;
  using OSS = std::basic_osyncstream<CharT>;

  ASSERT_SAME_TYPE(typename Buf::char_type, CharT);
  ASSERT_SAME_TYPE(typename Buf::traits_type, std::char_traits<CharT>);
  ASSERT_SAME_TYPE(typename Buf::allocator_type, std::allocator<CharT>);

  ASSERT_SAME_TYPE(typename OSS::char_type, CharT);
  ASSERT_SAME_TYPE(typename OSS::traits_type, std::char_traits<CharT>);
  ASSERT_SAME_TYPE(typename OSS::allocator_type, std::allocator<CharT>);
}

int main(int, char**) {
  ASSERT_SAME_TYPE(std::syncbuf, std::basic_syncbuf<char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  ASSERT_SAME_TYPE(std::wsyncbuf, std::basic_syncbuf<wchar_t>);
#endif
  ASSERT_SAME_TYPE(std::osyncstream, std::basic_osyncstream<char>);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  ASSERT_SAME_TYPE(std::wosyncstream, std::basic_osyncstream<wchar_t>);
#endif

  test_member_types<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_member_types<wchar_t>();
#endif

  return 0;
}
