//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <algorithm>

// UNSUPPORTED: c++03, c++11, c++14, c++17, c++20

// template<input_iterator I1, sentinel_for<I1> S1, input_iterator I2, sentinel_for<I2> S2,
//          class Pred = ranges::equal_to, class Proj1 = identity, class Proj2 = identity>
//   requires indirectly_comparable<I1, I2, Pred, Proj1, Proj2>
//   constexpr bool ranges::ends_with(I1 first1, S1 last1, I2 first2, S2 last2, Pred pred = {},
//                                      Proj1 proj1 = {}, Proj2 proj2 = {});
// template<input_range R1, input_range R2, class Pred = ranges::equal_to, class Proj1 = identity,
//          class Proj2 = identity>
//   requires indirectly_comparable<iterator_t<R1>, iterator_t<R2>, Pred, Proj1, Proj2>
//   constexpr bool ranges::ends_with(R1&& r1, R2&& r2, Pred pred = {},
//                                      Proj1 proj1 = {}, Proj2 proj2 = {});

#include <algorithm>
#include <array>
#include <chrono>
#include <ranges>
#include <vector>
#include "almost_satisfies_types.h"
#include "test_iterators.h"

using namespace std::chrono;

template <class Iter1, class Sent1 = Iter1, class Iter2 = int*, class Sent2 = Iter2>
concept HasEndsWithIt = requires(Iter1 first1, Sent1 last1, Iter2 first2, Sent2 last2) {
  std::ranges::ends_with(first1, last1, first2, last2);
};

static_assert(HasEndsWithIt<int*>);
static_assert(!HasEndsWithIt<ForwardIteratorNotDerivedFrom>);
static_assert(!HasEndsWithIt<ForwardIteratorNotIncrementable>);
static_assert(HasEndsWithIt<int*, int*>);
static_assert(!HasEndsWithIt<int*, SentinelForNotSemiregular>);
static_assert(!HasEndsWithIt<int*, int*, int**>); // not indirectly comparable
static_assert(!HasEndsWithIt<int*, SentinelForNotWeaklyEqualityComparableWith>);
static_assert(!HasEndsWithIt<int*, int*, ForwardIteratorNotDerivedFrom>);
static_assert(!HasEndsWithIt<int*, int*, ForwardIteratorNotIncrementable>);
static_assert(!HasEndsWithIt<int*, int*, int*, SentinelForNotSemiregular>);
static_assert(!HasEndsWithIt<int*, int*, int*, SentinelForNotWeaklyEqualityComparableWith>);

template <class Range1, class Range2 = UncheckedRange<int*>>
concept HasEndsWithR = requires(Range1 range1, Range2 range2) { std::ranges::ends_with(range1, range2); };

static_assert(HasEndsWithR<UncheckedRange<int*>>);
static_assert(!HasEndsWithR<ForwardRangeNotDerivedFrom>);
static_assert(!HasEndsWithR<ForwardIteratorNotIncrementable>);
static_assert(!HasEndsWithR<ForwardRangeNotSentinelSemiregular>);
static_assert(!HasEndsWithR<ForwardRangeNotSentinelEqualityComparableWith>);
static_assert(!HasEndsWithR<UncheckedRange<int*>, UncheckedRange<int**>>); // not indirectly comparable
static_assert(!HasEndsWithR<UncheckedRange<int*>, ForwardRangeNotDerivedFrom>);
static_assert(!HasEndsWithR<UncheckedRange<int*>, ForwardRangeNotIncrementable>);
static_assert(!HasEndsWithR<UncheckedRange<int*>, ForwardRangeNotSentinelSemiregular>);
static_assert(!HasEndsWithR<UncheckedRange<int*>, ForwardRangeNotSentinelEqualityComparableWith>);

// clang-format off
template <class Iter1, class Sent1 = Iter1, class Iter2, class Sent2 = Iter2>
constexpr void test_iterators() {
  { // simple tests
    int a[]          = {1, 2, 3, 4, 5, 6};
    int p[]          = {4, 5, 6};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
    auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      [[maybe_unused]] std::same_as<bool> decltype(auto) ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
      assert(ret);
    }
    {
      [[maybe_unused]] std::same_as<bool> decltype(auto) ret = std::ranges::ends_with(whole, suffix);
      assert(ret);
    }
  }

  { // suffix doesn't match
    int a[] = {1, 2, 3, 4, 5, 6};
    int p[] = {1, 2, 3};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
    auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
    {
      bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::ends_with(whole, suffix);
      assert(!ret);
    }
  }

  { // range consists of just one element
    int a[] = {1};
    int p[] = {1};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 1)));
    auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 1)));
    {
      bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::ends_with(whole, suffix);
      assert(ret);
    }
  }

  { // suffix consists of just one element
    int a[] = {5, 1, 2, 4, 3};
    int p[] = {3};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 5)));
    auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 1)));
    {
      bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::ends_with(whole, suffix);
      assert(ret);
    }
  }

  { // range and suffix are identical
    int a[] = {1, 2, 3, 4, 5, 6};
    int p[] = {1, 2, 3, 4, 5, 6};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
    auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 6)));
    {
      bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
      assert(ret);
    }
    {
      bool ret = std::ranges::ends_with(whole, suffix);
      assert(ret);
    }
  }

  { // suffix is longer than range
    int a[] = {3, 4, 5, 6, 7, 8};
    int p[] = {1, 2, 3, 4, 5, 6, 7, 8};
    auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
    auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 8)));
    {
      bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
      assert(!ret);
    }
    {
      bool ret = std::ranges::ends_with(whole, suffix);
      assert(!ret);
    }
 }

 { // suffix has zero length
   int a[] = {1, 2, 3, 4, 5, 6};
   int p[] = {};
   auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
   auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p)));
   {
     bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
     assert(ret);
   }
   {
     bool ret = std::ranges::ends_with(whole, suffix);
     assert(ret);
   }
 }

 { // range has zero length
   int a[] = {};
   int p[] = {1, 2, 3, 4, 5, 6, 7, 8};
   auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a)));
   auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 8)));
   {
     bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
     assert(!ret);
   }
   {
     bool ret = std::ranges::ends_with(whole, suffix);
     assert(!ret);
   }
 }

 { // subarray
   int a[] = {0, 3, 5, 10, 7, 3, 5, 89, 3, 5, 2, 1, 8, 6};
   int p[] = {3, 5};
   auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 13)));
   auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 2)));
   {
     bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
     assert(!ret);
   }
   {
     bool ret = std::ranges::ends_with(whole, suffix);
     assert(!ret);
   }
 }

 { // repeated suffix
   int a[] = {8, 6, 3, 5, 1, 2};
   int p[] = {1, 2, 1, 2};
   auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
   auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 4)));
   {
     bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end());
     assert(!ret);
   }
   {
     bool ret = std::ranges::ends_with(whole, suffix);
     assert(!ret);
   }
 }

 { // check that the predicate is used
   int a[] = {5, 1, 3, 2, 7};
   int p[] = {-2, -7};
   auto pred = [](int l, int r) { return l * -1 == r; };
   auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 5)));
   auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 2)));
   {
     bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end(), pred);
     assert(ret);
   }
   {
     bool ret = std::ranges::ends_with(whole, suffix, pred);
     assert(ret);
   }
 }

 { // check that the projections are used
   int a[] = {1, 3, 15, 1, 2, 1};
   int p[] = {2, 1, 2};
   auto whole = std::ranges::subrange(Iter1(a), Sent1(Iter1(a + 6)));
   auto suffix  = std::ranges::subrange(Iter2(p), Sent2(Iter2(p + 3)));
   {
     bool ret = std::ranges::ends_with(whole.begin(), whole.end(), suffix.begin(), suffix.end(), {},
         [](int i) { return i - 3; },
         [](int i) { return i * -1; });
     assert(ret);
   }
   {
     bool ret = std::ranges::ends_with(whole, suffix, {},
         [](int i) { return i - 3; },
         [](int i) { return i * -1; });
     assert(ret);
   }
  }
}

constexpr bool test() {
  // This is to test (forward_iterator<_Iter1> || sized_sentinel_for<_Sent1, _Iter1>) condition.
  types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iter2>() {
    types::for_each(types::cpp20_input_iterator_list<int*>{}, []<class Iter1>() {
      if constexpr (std::forward_iterator<Iter1> && std::forward_iterator<Iter2>)
        test_iterators<Iter1, Iter1, Iter2, Iter2>();
      if constexpr (std::forward_iterator<Iter2>)
        test_iterators<Iter1, sized_sentinel<Iter1>, Iter2, Iter2>();
      if constexpr (std::forward_iterator<Iter1>)
        test_iterators<Iter1, Iter1, Iter2, sized_sentinel<Iter2>>();
      test_iterators<Iter1, sized_sentinel<Iter1>, Iter2, sized_sentinel<Iter2>>();
    });
  });

  return true;
}

int main(int, char**) {
  test();
  static_assert(test());

  { // benmark test to check that random access optimization
    std::vector<int> a = {100, 32};
    std::vector<int> p = {32};
    for (int i = 0; i < 2147483640; i++) {
      p.push_back(i);
      a.push_back(i);
    }
    auto start = high_resolution_clock::now();
    { // test bidirectional algorithm as a benmark
      auto __rbegin1 = std::make_reverse_iterator(a.end());
      auto __rend1   = std::make_reverse_iterator(a.begin());
      auto __rbegin2 = std::make_reverse_iterator(p.end());
      auto __rend2   = std::make_reverse_iterator(p.begin());
      bool ret = std::ranges::starts_with(__rbegin1, __rend1, __rbegin2, __rend2);
      assert(ret);
    }
    auto stop = high_resolution_clock::now();
    auto bidirectional_duration = duration_cast<microseconds>(stop - start);
    start = high_resolution_clock::now();
    { // test random access optimization
      auto __n1 = std::ranges::distance(a.begin(), a.end());
      auto __n2 = std::ranges::distance(p.begin(), p.end());
      if (__n2 == 0)
        return true;
      if (__n2 > __n1)
        return false;
      auto __offset = __n1 - __n2;
      std::vector<int>::iterator i1 = a.begin();
      std::ranges::advance(i1, __offset);
      bool ret = std::ranges::equal(i1, a.end(), p.begin(), p.end(), {}, {}, {});
      assert(ret);
    }
    stop = high_resolution_clock::now();
    auto random_duration = duration_cast<microseconds>(stop - start);
    // verify that random access optimization takes less time
    assert(bidirectional_duration > random_duration);
  }
  return 0;
}
