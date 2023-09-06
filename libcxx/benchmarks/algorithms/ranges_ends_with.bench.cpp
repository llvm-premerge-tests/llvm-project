//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include <algorithm>
#include <benchmark/benchmark.h>
#include <iterator>
#include <list>

static void bm_ends_with_random_iter(benchmark::State& state) {
  std::list<int> a(state.range(), 1);
  std::list<int> p(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(p);
    auto begin1 = std::random_access_iterator<decltype(a.begin())>;
    auto end1 = std::random_access_iterator<decltype(a.end())>;
    auto begin2 = std::random_access_iterator<decltype(p.begin())>;
    auto end2 = std::random_access_iterator<decltype(p.end())>;
    std::list<int> newa(begin1, end1);
    std::list<int> newp(begin2, end2);
    benchmark::DoNotOptimize(std::ranges::ends_with(newa, newp));
  }
}
BENCHMARK(bm_ends_with_random_iter)->RangeMultiplier(16)->Range(16, 16<<20);

static void bm_ends_with_bidirectional_iter(benchmark::State& state) {
  std::list<int> a(state.range(), 1);
  std::list<int> p(state.range(), 1);

  for (auto _ : state) {
    benchmark::DoNotOptimize(a);
    benchmark::DoNotOptimize(p);
    auto begin1 = std::bidirectional_iterator<decltype(a.begin())>;
    auto end1 = std::bidirectional_iterator<decltype(a.end())>;
    auto begin2 = std::bidirectional_iterator<decltype(p.begin())>;
    auto end2 = std::bidirectional_iterator<decltype(p.end())>;
    std::list<int> newa(begin1, end1);
    std::list<int> newp(begin2, end2);
    benchmark::DoNotOptimize(std::ranges::ends_with(newa, newp));
  }
}
BENCHMARK(bm_ends_with_bidirectional_iter)->RangeMultiplier(16)->Range(16, 16<<20);

BENCHMARK_MAIN();
