//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// <memory>

// [out.ptr.t], class template out_ptr_t
// template<class Smart, class Pointer, class... Args>
//   class out_ptr_t;                                          // since c++23

#include <memory>

int main(int, char**) {
  {
    std::unique_ptr<int> uPtr;

    std::out_ptr_t<std::unique_ptr<int>, int*>{uPtr};
  }
  {
    std::unique_ptr<int, std::default_delete<int>> uPtr;

    std::out_ptr_t<decltype(uPtr), int*, std::default_delete<int>>{uPtr, std::default_delete<int>{}};
  }

  return 0;
}
