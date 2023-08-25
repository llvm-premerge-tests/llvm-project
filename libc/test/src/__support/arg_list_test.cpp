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

long int check_primitives(int first, ...) {
  va_list vlist;
  va_start(vlist, first);
  __llvm_libc::internal::ArgList args(vlist);
  va_end(vlist);

  long int count = 0;
  count += args.next_var<int>();
  count += args.next_var<int>();
  count += args.next_var<int>();
  count += args.next_var<long>();
  count += args.next_var<long>();
  count += args.next_var<intmax_t>();
  count += args.next_var<size_t>();
  count += args.next_var<ptrdiff_t>();
  count += args.next_var<double>();
  count += args.next_var<double>();
  count += args.next_var<long double>();
  count += *args.next_var<int *>();
  return count;
}

TEST(LlvmLibcArgListTest, TestPrimitiveTypes) {
  char c = 1;
  short s = 1;
  int i = 1;
  long l = 1;
  long long ll = 1;
  intmax_t m = 1;
  size_t st = 1;
  ptrdiff_t pt = 1;
  float f = 1.0;
  double d = 1.0;
  long double ld = 1.0;
  void *p = &i;
  ASSERT_EQ(check_primitives(0, c, s, i, l, ll, m, st, pt, f, d, ld, p), 12l);
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

// Test vector extensions from clang.
#ifdef __clang__

using int1 = int __attribute__((ext_vector_type(1)));
using int2 = int __attribute__((ext_vector_type(2)));
using int3 = int __attribute__((ext_vector_type(3)));
using int4 = int __attribute__((ext_vector_type(4)));

int check_vector_type(int first, ...) {
  va_list vlist;
  va_start(vlist, first);
  __llvm_libc::internal::ArgList args(vlist);
  va_end(vlist);

  int1 v1 = args.next_var<int1>();
  int2 v2 = args.next_var<int2>();
  int3 v3 = args.next_var<int3>();
  int4 v4 = args.next_var<int4>();

  return v1.x + v2.x + v2.y + v3.x + v3.y + v3.z + v4.x + v4.y + v4.z + v4.w;
}

TEST(LlvmLibcArgListTest, TestVectorTypes) {
  int1 v1 = {1};
  int2 v2 = {1, 2};
  int3 v3 = {1, 2, 3};
  int4 v4 = {1, 2, 3, 4};
  ASSERT_EQ(check_vector_type(0, v1, v2, v3, v4), 20);
}

#endif
