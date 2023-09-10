//===----------------------------------------------------------------------===//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20
// UNSUPPORTED: no-filesystem
// UNSUPPORTED: GCC-ALWAYS_INLINE-FIXME

// XFAIL: availability-fp_to_chars-missing
// XFAIL: availability-print-missing

// Clang modules do not work with the definiton of _LIBCPP_TESTING_PRINT_IS_TERMINAL
// XFAIL: modules-build
// <ostream>

// Tests the implementation of
//  void __vprint_unicode(ostream& os, string_view fmt,
//                        format_args args, bool write_nl);

// In the library when the std::cout is redirected to a file it is no
// longer considered a terminal and the special terminal handling is no
// longer executed. By testing this function we can "force" emulate a
// terminal.
// Note write_nl is tested by the public API.

#include <cstdio>
bool is_terminal(FILE*);
#define _LIBCPP_TESTING_PRINT_IS_TERMINAL ::is_terminal

#include "filesystem_test_helper.h"
#include <cassert>
#include <fstream>
#include <iostream>
#include <ostream>
#include <sstream>

#include "test_macros.h"

scoped_test_env env;
std::string filename = env.create_file("output.txt");

int is_terminal_calls        = 0;
bool should_call_is_terminal = false;
bool is_terminal_result      = false;
bool is_terminal(FILE*) {
  ++is_terminal_calls;
  assert(should_call_is_terminal);
  return is_terminal_result;
}

static void test_is_terminal_not_a_file_stream() {
  is_terminal_calls       = 0;
  should_call_is_terminal = false;
  is_terminal_result      = false;
  {
    std::stringstream stream;
    std::print(stream, "test");
  }
  {
    std::ostringstream stream;
    std::print(stream, "test");
  }
  assert(is_terminal_calls == 0);
}

static void test_is_terminal_file_stream() {
  is_terminal_calls       = 0;
  should_call_is_terminal = true;
  is_terminal_result      = false;
  {
    std::fstream stream(filename);
    assert(stream.is_open());
    assert(stream.good());
    std::print(stream, "test");
    assert(is_terminal_calls == 1);
  }
  {
    std::ofstream stream(filename);
    assert(stream.is_open());
    assert(stream.good());
    std::print(stream, "test");
    assert(is_terminal_calls == 2);
  }
}

static void test_is_terminal_rdbuf_derived_from_filebuf() {
  struct my_filebuf : public std::filebuf {};

  is_terminal_calls       = 0;
  should_call_is_terminal = true;
  is_terminal_result      = false;

  my_filebuf buf;
  buf.open(filename, std::ios_base::out);
  assert(buf.is_open());

  std::ostream stream(&buf);
  std::print(stream, "test");
  assert(is_terminal_calls == 1);
}

static void test_is_terminal_std_cout_cerr_clog() {
  is_terminal_calls       = 0;
  should_call_is_terminal = true;
  is_terminal_result      = false;
  {
    std::print(std::cout, "test");
    assert(is_terminal_calls == 1);
  }
  {
    std::print(std::cerr, "test");
    assert(is_terminal_calls == 2);
  }
  {
    std::print(std::clog, "test");
    assert(is_terminal_calls == 3);
  }
}

static void test_is_terminal_is_flushed() {
  struct sync_counter : public std::filebuf {
    sync_counter() {
      open(filename, std::ios_base::out);
      assert(is_open());
    }
    int sync_calls = 0;

  protected:
    int virtual sync() {
      ++sync_calls;
      return std::basic_streambuf<char>::sync();
    }
  };

  should_call_is_terminal = true;
  is_terminal_result      = false;

  sync_counter buf;
  std::ostream stream(&buf);

  // Not a terminal sync is not called.
  std::print(stream, "");
  assert(buf.sync_calls == 0);

  // A terminal sync is called.
  is_terminal_result = true;
  std::print(stream, "");
  assert(buf.sync_calls == 1); // only called from the destructor of the sentry
}

#ifdef _WIN32
static void test_transcoding() {
  // TODO PRINT Investigate on Windows why it fails.
#  if 0
  is_terminal_calls       = 0;
  should_call_is_terminal = true;
  is_terminal_result      = true;

  {
    std::ofstream stream(filename);
    // 2 code units UTF-8 0xCC 0x81, should be transcoded to 1 code unit UTF-16 0x0301
    std::print(stream, "\u0301");
    assert(is_terminal_calls == 1);
  }


  FILE* file = std::fopen(filename.c_str(), "r");
  assert(std::fgetc(file) == 0x03);
  assert(std::fgetc(file) == 0x01);
  assert(std::fgetc(file) == EOF);
  std::fclose(file);
#  endif
}
#endif

int main(int, char**) {
  test_is_terminal_not_a_file_stream();
  test_is_terminal_file_stream();
  test_is_terminal_rdbuf_derived_from_filebuf();
  test_is_terminal_std_cout_cerr_clog();

  test_is_terminal_is_flushed();

#ifdef _WIN32
  test_transcoding();
#endif

  return 0;
}
