//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// REQUIRES: c++11 || c++14 || c++17

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCPP_DISABLE_DEPRECATION_WARNINGS

// <functional>

// reference_wrapper

// check for types of class reference_wrapper:
//   argument_type, first_argument_type, second_argument_type
// C++11 20.8.3 [refwrap]

#include <functional>
#include <type_traits>

void test() {
  {
    typedef std::reference_wrapper<int>::argument_type A; // expected-error {{no type named}}
    typedef std::reference_wrapper<int>::result_type R; // expected-error {{no type named}}
    typedef std::reference_wrapper<int>::first_argument_type A1; // expected-error {{no type named}}
    typedef std::reference_wrapper<int>::second_argument_type A2; // expected-error {{no type named}}
  }
  {
    typedef float(F)(int);
    typedef std::reference_wrapper<F>::argument_type A;
    typedef std::reference_wrapper<F>::result_type R;
    static_assert(std::is_same<A, int>::value, "");
    static_assert(std::is_same<R, float>::value, "");
  }
  {
    typedef double(F)(char, int);
    typedef std::reference_wrapper<F>::first_argument_type A1;
    typedef std::reference_wrapper<F>::second_argument_type A2;
    typedef std::reference_wrapper<F>::result_type R;
    static_assert(std::is_same<A1, char>::value, "");
    static_assert(std::is_same<A2, int>::value, "");
    static_assert(std::is_same<R, double>::value, "");
  }
  {
    struct S {};
    typedef long (S::*P)(void);
    typedef std::reference_wrapper<P>::argument_type A;
    static_assert(std::is_same<A, S*>::value, "");
  }
  {
    struct S {};
    typedef void (S::*P)(int);
    typedef std::reference_wrapper<P>::first_argument_type A1;
    typedef std::reference_wrapper<P>::second_argument_type A2;
    static_assert(std::is_same<A1, S*>::value, "");
    static_assert(std::is_same<A2, int>::value, "");
  }
  {
    struct S {
      typedef int argument_type;
    };
    typedef std::reference_wrapper<S>::argument_type A;
    static_assert(std::is_same<S::argument_type, A>::value, "");
  }
  {
    struct S {
      typedef char first_argument_type;
      typedef int second_argument_type;
    };
    typedef std::reference_wrapper<S>::first_argument_type A1;
    typedef std::reference_wrapper<S>::second_argument_type A2;
    static_assert(std::is_same<S::first_argument_type, A1>::value, "");
    static_assert(std::is_same<S::second_argument_type, A2>::value, "");
  }
}
