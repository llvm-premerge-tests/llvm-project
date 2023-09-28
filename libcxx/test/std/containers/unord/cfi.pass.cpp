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

#include <cassert>
#include <unordered_map>
#include <unordered_set>

int called = 0;

void increment1() { called++; }
void increment2() { called++; }
void increment3() { called++; }

int main(int, char**) {
  using Fptr = void (*)();
  {
    called = 0;
    std::unordered_set<Fptr> c;
    c.insert(increment1);
    c.insert(increment2);
    c.insert(increment3);
    for (auto f : c) {
      f();
    }
    assert(called == 3);
  }
  {
    called = 0;
    std::unordered_multiset<Fptr> c;
    c.insert(increment1);
    c.insert(increment2);
    c.insert(increment3);
    for (auto f : c) {
      f();
    }
    assert(called == 3);
  }
  {
    called = 0;
    std::unordered_map<int, Fptr> c;
    c.insert({1, increment1});
    c.insert({2, increment2});
    c.insert({3, increment3});
    for (auto f : c) {
      f.second();
    }
    assert(called == 3);
  }
  {
    called = 0;
    std::unordered_multimap<int, Fptr> c;
    c.insert({1, increment1});
    c.insert({2, increment2});
    c.insert({3, increment3});
    for (auto f : c) {
      f.second();
    }
    assert(called == 3);
  }

  return 0;
}
