//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This should work for modules too, but clang doesn't emit the diagnostics for unknown reasons.
// XFAIL: modules-build && clang-16
// XFAIL: modules-build && clang-17
// XFAIL: modules-build && apple-clang-14
// XFAIL: modules-build && apple-clang-15

// <string>

// template<> struct char_traits<T> for arbitrary T

// Make sure we issue deprecation warnings.

#include <string>

void f() {
    std::char_traits<unsigned char> t1; (void)t1; // expected-warning{{'char_traits<unsigned char>' is deprecated}}
    std::char_traits<signed char> t2; (void)t2; // expected-warning{{'char_traits<signed char>' is deprecated}}
    std::char_traits<unsigned long> t3; (void)t3; // expected-warning{{'char_traits<unsigned long>' is deprecated}}
}
