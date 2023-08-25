//===-- Unittests for ArgList ---------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "src/__support/arg_list.h"

#include "test/UnitTest/Test.h"

int get_nth_int(int n, ...) {
  va_list vlist;
  va_start(vlist, n);
  __llvm_libc::internal::ArgList v(vlist);
  va_end(vlist);

  for (int i = 0; i < n; ++i) {
    v.next_var<int>();
  }
  return v.next_var<int>();
}

TEST(LlvmLibcArgListTest, BasicUsage) {
  ASSERT_EQ(get_nth_int(5, 0, 10, 20, 30, 40, 50, 60, 70, 80, 90), 50);
}

int sum_two_nums(int first, int second, ...) {
  va_list vlist;
  va_start(vlist, second);
  __llvm_libc::internal::ArgList v1(vlist);
  va_end(vlist);

  __llvm_libc::internal::ArgList v2 = v1;

  int first_val;
  for (int i = 0; i < first; ++i) {
    v1.next_var<int>();
  }
  first_val = v1.next_var<int>();

  int second_val;
  for (int i = 0; i < second; ++i) {
    v2.next_var<int>();
  }
  second_val = v2.next_var<int>();

  return first_val + second_val;
}

TEST(LlvmLibcArgListTest, CopyConstructor) {
  ASSERT_EQ(sum_two_nums(3, 1, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
            10);

  ASSERT_EQ(sum_two_nums(3, 5, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024),
            40);
}

bool check_primitives(int first, ...) {
  va_list vlist;
  va_start(vlist, first);
  __llvm_libc::internal::ArgList args(vlist);
  va_end(vlist);

  if (args.next_var<int>() != 0)
    return false;
  if (args.next_var<int>() != 0)
    return false;
  if (args.next_var<int>() != 0)
    return false;
  if (args.next_var<long>() != 0)
    return false;
  if (args.next_var<long>() != 0)
    return false;
  if (args.next_var<intmax_t>() != 0)
    return false;
  if (args.next_var<size_t>() != 0)
    return false;
  if (args.next_var<ptrdiff_t>() != 0)
    return false;
  if (args.next_var<double>() != 0)
    return false;
  if (args.next_var<double>() != 0)
    return false;
  if (args.next_var<long>() != 0)
    return false;
  if (args.next_var<void *>() != 0)
    return false;
  return true;
}

TEST(LlvmLibcArgListTest, TestPrimitiveTypes) {
  char x1 = 0;
  short x2 = 0;
  int x3 = 0;
  long x4 = 0;
  long long x5 = 0;
  intmax_t x6 = 0;
  size_t x7 = 0;
  ptrdiff_t x8 = 0;
  float x9 = 0;
  double x10 = 0;
  long double x11 = 0;
  void *x12 = 0;
  ASSERT_TRUE(
      check_primitives(0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12));
}

struct S {
  char c;
  short s;
  int i;
  long l;
  float f;
  double d;
};

long int check_struct_type(int first, ...) {
  va_list vlist;
  va_start(vlist, first);
  __llvm_libc::internal::ArgList args(vlist);
  va_end(vlist);

  S s = args.next_var<S>();
  int last = args.next_var<int>();
  return s.c + s.s + s.i + s.l + s.f + s.d + last;
}

TEST(LlvmLibcArgListTest, TestStructTypes) {
  S s{'\x1', 1, 1, 1l, 1.0f, 1.0};
  ASSERT_EQ(check_struct_type(0, s, 1), 7l);
}
