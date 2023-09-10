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
#include <mutex>
#include <thread>
#include <cassert>
#include <iostream>

#include "test_macros.h"

using ms = std::chrono::milliseconds;
using OS = std::basic_osyncstream<char, std::char_traits<char>, std::allocator<char>>;

static std::basic_ostringstream<char> ss;
static const char a = 'a';
static const char b = 'b';
static const char c = 'c';
static const char d = 'd';

void f1() {
  OS out(ss);
  out << a;
  std::this_thread::sleep_for(ms(250));
  out << b;
}

void f2() {
  OS out(ss);
  out << c;
  out << d;
}

void test_mutlithread() {
  std::thread t1(f1);
  std::thread t2(f2);
  t1.join();
  t2.join();

  assert(ss.str() == "cdab");
}

int main(int, char**) {
  // tested with char only for now
  test_mutlithread();

  return 0;
}
