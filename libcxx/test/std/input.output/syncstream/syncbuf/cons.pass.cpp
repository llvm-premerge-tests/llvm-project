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
struct test_allocator : std::allocator<T> {
  int id;
  int moves  = 0;
  int copies = 0;

  test_allocator(int id = 0) : id(id) {}
  test_allocator(test_allocator const& other) : id(other.id), moves(other.moves), copies(other.copies) {}
  test_allocator(test_allocator&& other)
      : id(std::move(other.id)), moves(std::move(other.moves)), copies(std::move(other.copies)) {}
};

template <class T>
class test_buf : public std::basic_streambuf<T> {};

template <class CharT>
void test_streambuf_const() {
  using BuffT = std::basic_syncbuf<CharT, std::char_traits<CharT>, test_allocator<CharT>>;

  test_buf<CharT> base;
  test_allocator<CharT> alloc(42);
  BuffT buff(&base, alloc);

  assert(buff.get_wrapped() == &base);
  assert(buff.get_allocator().id == alloc.id);
}

template <class CharT>
void test_move_const() {
  using BuffT = std::basic_syncbuf<CharT, std::char_traits<CharT>, test_allocator<CharT>>;

  test_buf<CharT> base;
  test_allocator<CharT> alloc(42);
  BuffT buff1(&base, alloc);
  BuffT buff2(std::move(buff1));

  assert(buff1.get_wrapped() == nullptr);
  assert(buff2.get_wrapped() == &base);

  // LWG issue?
  // assert(buff1.rdbuf()->pbase() == buff1.rdbuf()->pptr());
}

int main(int, char**) {
  test_streambuf_const<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_streambuf_const<wchar_t>();
#endif

  test_move_const<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_move_const<wchar_t>();
#endif

  return 0;
}
