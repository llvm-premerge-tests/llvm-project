//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
#define TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H

#include <ranges>

struct Range : std::ranges::view_base {
  constexpr explicit Range(int* b, int* e) : begin_(b), end_(e) {}
  constexpr Range(Range const& other) : begin_(other.begin_), end_(other.end_), wasCopyInitialized(true) {}
  constexpr Range(Range&& other) : begin_(other.begin_), end_(other.end_), wasMoveInitialized(true) {}
  Range& operator=(Range const&) = default;
  Range& operator=(Range&&)      = default;
  constexpr int* begin() const { return begin_; }
  constexpr int* end() const { return end_; }

  int* begin_;
  int* end_;
  bool wasCopyInitialized = false;
  bool wasMoveInitialized = false;
};

struct NoCopyRange : std::ranges::view_base {
  explicit NoCopyRange(int*, int*);
  NoCopyRange(NoCopyRange const&)            = delete;
  NoCopyRange(NoCopyRange&&)                 = default;
  NoCopyRange& operator=(NoCopyRange const&) = default;
  NoCopyRange& operator=(NoCopyRange&&)      = default;
  int* begin() const;
  int* end() const;
};

#endif // TEST_STD_RANGES_RANGE_ADAPTORS_RANGE_STRIDE_TYPES_H
