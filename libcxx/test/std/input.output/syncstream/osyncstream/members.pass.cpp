//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14 ,c++17

// <syncstream>

// template <class charT, class traits, class Allocator>
// class basic_osyncstream;

// void emit();
// streambuf_type* get_wrapped() const noexcept;

#include <syncstream>
#include <sstream>
#include <cassert>

#include "test_macros.h"

template <class T>
class test_buf : public std::basic_streambuf<T> {};

template <class CharT>
void test_emit() {
  // XXX exceptions

  using OS = std::basic_osyncstream<CharT, std::char_traits<CharT>, std::allocator<CharT>>;
  using SS = std::basic_ostringstream<CharT>;
  CharT c  = 'f';

  SS ss;
  OS out(ss);
  out << c;
  assert(ss.str().empty());
  out.emit();
  assert(ss.str()[0] == c);
}

template <class CharT>
void test_get_wrapped() {
  using OS = std::basic_osyncstream<CharT, std::char_traits<CharT>, std::allocator<CharT>>;

  test_buf<CharT> base;
  std::allocator<CharT> alloc;
  OS out(&base, alloc);
  assert(out.get_wrapped() == &base);
  ASSERT_NOEXCEPT(out.get_wrapped());
}

template <class CharT>
void test_rdbuf() {
  using OS = std::basic_osyncstream<CharT, std::char_traits<CharT>, std::allocator<CharT>>;

  test_buf<CharT> base;
  std::allocator<CharT> alloc;
  OS out(&base, alloc);
  assert(out.rdbuf() != nullptr);
  ASSERT_SAME_TYPE(decltype(out.rdbuf()), std::basic_syncbuf<CharT, std::char_traits<CharT>, std::allocator<CharT>>*);
  ASSERT_NOEXCEPT(out.rdbuf());
}

int main(int, char**) {
  test_emit<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_emit<wchar_t>();
#endif

  test_get_wrapped<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_get_wrapped<wchar_t>();
#endif

  test_rdbuf<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_rdbuf<wchar_t>();
#endif

  return 0;
}
