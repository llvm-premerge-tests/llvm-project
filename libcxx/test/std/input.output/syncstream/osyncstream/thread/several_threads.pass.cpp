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

#include <cassert>
#include <chrono>
#include <mutex>
#include <sstream>
#include <string>
#include <syncstream>
#include <thread>
#include <unordered_set>
#include <vector>

#include "test_macros.h"

using OS = std::basic_osyncstream<char, std::char_traits<char>, std::allocator<char>>;

static std::basic_ostringstream<char> ss;
static std::unordered_multiset<std::string> test_strings = {
    "C++",
    "is",
    "a",
    "general-purpose",
    "programming",
    "language",
    "created",
    "by",
    "Bjarne",
    "Stroustrup",
    "as",
    "an",
    "extension",
    "of",
    "the",
    "C",
    "programming",
    "language,",
    "or",
    "C",
    "with",
    "Classes",
    "The",
    "language",
    "has",
    "expanded",
    "significantly",
    "over",
    "time,",
    "and",
    "modern",
    "C++",
    "has",
    "object-oriented,",
    "generic,",
    "and",
    "functional",
    "features",
    "in",
    "addition",
    "to",
    "facilities",
    "for",
    "low-level",
    "memory",
    "manipulation.",
    "It",
    "is",
    "almost",
    "always",
    "implemented",
    "as",
    "a",
    "compiled",
    "language,",
    "and",
    "many",
    "vendors",
    "provide",
    "C++",
    "compilers,",
    "including",
    "the",
    "Free",
    "Software",
    "Foundation,",
    "LLVM,",
    "Microsoft,",
    "Intel,",
    "and",
    "IBM,",
    "so",
    "it",
    "is",
    "available",
    "on",
    "many",
    "platforms."};

void f(std::string text) {
  OS out(ss);
  out << text;
}

void test() {
  ss = std::basic_ostringstream<char>();
  std::vector<std::thread> threads;
  for (std::string const& word : test_strings)
    threads.push_back(std::thread(f, word));
  for (size_t i = 0; i < test_strings.size(); (void)++i)
    threads[i].join();

  std::string output = ss.str();
  for (std::string const& word : test_strings)
    assert(output.find(word) != std::string::npos);
}

static std::basic_ostringstream<char> ss2;

void simple_fn() {
  OS out(ss2);
  out << "X";
}

unsigned measure_average_operation(unsigned thread_count) {
  std::vector<std::thread> threads;
  threads.reserve(thread_count);
  auto start = std::chrono::high_resolution_clock::now();
  for (unsigned i = 0; i < thread_count; ++i)
    threads.push_back(std::thread(simple_fn));
  for (unsigned i = 0; i < thread_count; ++i)
    threads[i].join();
  auto end            = std::chrono::high_resolution_clock::now();
  auto total_duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();
  return total_duration / thread_count;
}

// TODO: this isn't a great test because the overhead of everything else if far more than any
// performance issues syncstream might cause.
void test_constant_performance() {
  // Make sure that doing 10x the work takes no longer than 10x the time.
  unsigned ten_threads      = measure_average_operation(10);
  unsigned hundred_threads  = measure_average_operation(100);
  unsigned thousand_threads = measure_average_operation(1000);
  assert(ten_threads * 10 > hundred_threads);
  assert(hundred_threads * 10 > thousand_threads);
}

int main(int, char**) {
  // The more we test, the more likely we catch an error
  for (size_t i = 0; i < 10000; ++i)
    test();

  // Test that the complexity is linear (adding x threads takes roughly x times longer).
  test_constant_performance();

  return 0;
}
