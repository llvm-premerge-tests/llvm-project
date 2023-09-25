//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03
// UNSUPPORTED: availability-filesystem-missing
// UNSUPPORTED: no-localization, no-wide-characters

// <filesystem>

// class path

// template <class Source>
// path(const Source& source, const locale& loc, format = format::auto_format);
//
// template <class InputIterator>
// path(InputIterator first, InputIterator last, const locale& loc, format = format::auto_format);

#include "filesystem_include.h"
#include <cassert>
#include <cstddef>
#include <locale>
#include <type_traits>

#include "../../path_helper.h"
#include "test_macros.h"
#include "test_iterators.h"
#include "min_allocator.h"

template <class... Args>
void RunTestCase(const char* TestPath, const char* Expect, const std::locale& Locale, Args... args) {
  const char* TestPathEnd = StrEnd(TestPath);
  const std::size_t Size  = TestPathEnd - TestPath;
  const std::size_t SSize = StrEnd(Expect) - Expect;
  assert(Size == SSize);
  // StringTypes
  {
    const std::string S(TestPath);
    fs::path p(S, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  {
    const std::string_view S(TestPath);
    fs::path p(S, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  // Char* pointers
  {
    fs::path p(TestPath, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  {
    fs::path p(TestPath, TestPathEnd, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  // Iterators
  {
    using It = cpp17_input_iterator<const char*>;
    fs::path p(It{TestPath}, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
  {
    using It = cpp17_input_iterator<const char*>;
    fs::path p(It{TestPath}, It{TestPathEnd}, Locale, args...);
    assert(p.native() == Expect);
    assert(p.string<char>() == Expect);
  }
}

void test_sfinae() {
  {
    using It = cpp17_output_iterator<char*>;
    static_assert(!std::is_constructible<fs::path, It, std::locale>::value, "");
  }
  {
    using It = int*;
    static_assert(!std::is_constructible<fs::path, It, std::locale>::value, "");
  }
}

struct CustomCodeCvt : std::codecvt<wchar_t, char, std::mbstate_t> {
protected:
  result do_in(state_type&,
               const extern_type* from,
               const extern_type* from_end,
               const extern_type*& from_next,
               intern_type* to,
               intern_type* to_end,
               intern_type*& to_next) const override {
    for (; from < from_end && to < to_end; ++from, ++to)
      *to = 'o';

    from_next = from;
    to_next   = to;

    return result::ok;
  }
};

int main(int, char**) {
  std::locale Locale;

  // Ensure std::codecvt<wchar_t, char, std::mbstate_t> is used.
  std::locale CustomLocale(Locale, new CustomCodeCvt());
  std::string TestPath("aaaa");
  std::string Expect("oooo");
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale);
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale, fs::path::format::auto_format);
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale, fs::path::format::native_format);
  RunTestCase(TestPath.c_str(), Expect.c_str(), CustomLocale, fs::path::format::generic_format);

  for (auto const& MS : PathList) {
    RunTestCase(MS, MS, Locale);
    RunTestCase(MS, MS, Locale, fs::path::format::auto_format);
    RunTestCase(MS, MS, Locale, fs::path::format::native_format);
    RunTestCase(MS, MS, Locale, fs::path::format::generic_format);
  }

  test_sfinae();

  return 0;
}
