//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17
// UNSUPPORTED: no-localization

// <syncstream>

// template <class charT, class traits, class Allocator>
// class basic_syncbuf;

// void set_emit_on_sync(bool) noexcept;

#include <syncstream>
#include <cassert>

#include "test_macros.h"
#include "helpers.h"

template <class T>
void test_set_emit_on_sync() {
  // set_emit_on_sync tested in sync
  test_syncbuf<T, std::allocator<T>> buff(nullptr, std::allocator<T>());
  ASSERT_NOEXCEPT(buff.set_emit_on_sync(false));
}

int main(int, char**) {
  test_set_emit_on_sync<char>();
#ifndef TEST_HAS_NO_WIDE_CHARACTERS
  test_set_emit_on_sync<wchar_t>();
#endif

  return 0;
}
