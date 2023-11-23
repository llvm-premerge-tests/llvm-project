//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++17
// <functional>
// Types of class reference_wrapper were removed in C++20

// Check that types of class reference_wrapper are marked deprecated

#include <functional>

typedef float(F1)(int);
typedef double(F2)(char, int);

std::reference_wrapper<F1>::argument_type a;         // expected-warning {{'argument_type' is deprecated}}
std::reference_wrapper<F1>::result_type r;           // expected-warning {{'result_type' is deprecated}}
std::reference_wrapper<F2>::first_argument_type a1;  // expected-warning {{'first_argument_type' is deprecated}}
std::reference_wrapper<F2>::second_argument_type a2; // expected-warning {{'second_argument_type' is deprecated}}
std::reference_wrapper<F2>::result_type r1;          // expected-warning {{'result_type' is deprecated}}
