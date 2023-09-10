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
// class basic_syncbuf;

// basic_syncbuf(streambuf_type* obuf, const Allocator& allocator);
// basic_syncbuf(basic_syncbuf&& other);

#include <syncstream>
#include <cassert>

#include "test_macros.h"

template <class T>
class test_streambuf : public std::basic_streambuf<T> {};

template <class CharT>
void test() {
  using BuffT = std::basic_syncbuf<CharT, std::char_traits<CharT>, std::allocator<CharT>>;

  test_streambuf<CharT> base1;
  test_streambuf<CharT> base2;
  std::allocator<CharT> alloc;
  BuffT buff1(&base1, alloc);
  BuffT buff2(&base2, alloc);
  std::swap(buff1, buff2);

  assert(buff1.get_wrapped() == &base2);
  assert(buff2.get_wrapped() == &base1);
}

int main(int, char**) {
  test<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test<wchar_t>();
#endif

  return 0;
}
