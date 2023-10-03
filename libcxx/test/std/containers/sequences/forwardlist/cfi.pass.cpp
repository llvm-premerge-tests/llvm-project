//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// This test attempts to trigger CFI violations if we were to store values inside
// container nodes in a murky way, such as using a char buffer instead of properly
// typed storage. See https://reviews.llvm.org/D101206 for details.

#include <forward_list>
#include <cassert>

int called = 0;

void increment() { called++; }

int main(int, char**) {
  using Fptr = void (*)();
  std::forward_list<Fptr> list;
  list.push_front(increment);
  list.push_front(increment);
  list.push_front(increment);
  for (auto f : list) {
    f();
  }
  assert(called == 3);

  return 0;
}
