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
#include <sstream>
#include <iostream>
#include <cassert>

#include "test_macros.h"

template <class T>
class test_buf : public std::basic_streambuf<T> {
public:
  T* _pbase() { return this->pbase(); }
};

template <class T>
class test_syncbuf : public std::basic_syncbuf<T, std::char_traits<T>, std::allocator<T>> {
public:
  test_syncbuf(test_buf<T>* buf, std::allocator<T> alloc)
      : std::basic_syncbuf<T, std::char_traits<T>, std::allocator<T>>(buf, alloc) {}

  int _sync() { return this->sync(); }
};

template <class CharT>
void test_sync(bool emit_on_sync) {
  std::basic_ostringstream<CharT> ss;
  test_buf<CharT>* base = reinterpret_cast<test_buf<CharT>*>(ss.rdbuf());
  std::allocator<CharT> alloc;
  test_syncbuf<CharT> buff(base, alloc);
  std::basic_ostream<CharT> out(&buff);

  buff.set_emit_on_sync(emit_on_sync);
  out << 'a';
  assert(base->_pbase() == nullptr);
  assert(buff._sync() == 0);

  if (emit_on_sync)
    assert(*base->_pbase() == 'a');
  else
    assert(base->_pbase() == nullptr);
}

int main(int, char**) {
  test_sync<char>(true);
  test_sync<char>(false);
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_sync<wchar_t>(true);
  test_sync<wchar_t>(false);
#endif

  return 0;
}
