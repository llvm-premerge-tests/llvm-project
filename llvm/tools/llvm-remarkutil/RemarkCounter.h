//===- RemarkCounter.h ----------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Generic tool to count remarks based on properties
//
//===----------------------------------------------------------------------===//
#ifndef TOOLS_LLVM_REMARKCOUNTER_H
#define TOOLS_LLVM_REMARKCOUNTER_H
#include "RemarkUtilHelpers.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Regex.h"
#include <list>

namespace llvm {
namespace remarks {

// Collect remarks by counting the existance of a remark or by looking through
// the keys and summing through the total count.
enum class CountBy { REMARK, KEYS };

// Summarize the count by either emitting one count for the remark file, or
// grouping the count by source file or by function name.
enum class GroupBy {
  TOTAL,
  PER_SOURCE,
  PER_FUNCTION,
  PER_FUNCTION_WITH_DEBUG_LOC
};

// Convert group by enums to string.
inline std::string groupByToStr(GroupBy GroupBy) {
  switch (GroupBy) {
  default:
    return "Total";
  case GroupBy::PER_FUNCTION:
    return "Function";
  case GroupBy::PER_FUNCTION_WITH_DEBUG_LOC:
    return "FuctionWithDebugLoc";
  case GroupBy::PER_SOURCE:
    return "Source";
  }
}

// Filter object which can be either a string or a regex to match with the
// remark properties.
struct FilterMatcher {
  std::variant<Regex, StringRef> FilterRE, FilterStr;
  bool IsRegex;
  FilterMatcher(std::string Filter, bool IsRegex) : IsRegex(IsRegex) {
    if (IsRegex)
      FilterRE = Regex(Filter);
    else
      FilterStr = StringRef(Filter);
  }
  bool match(StringRef StringToMatch) {
    if (IsRegex)
      return std::get<Regex>(FilterRE).match(StringToMatch);
    return std::get<StringRef>(FilterStr).equals(StringToMatch);
  }
};

/// Filter out remarks based on properties.
struct Filters {
  std::optional<FilterMatcher> RemarkNameFilter;
  std::optional<FilterMatcher> PassNameFilter;
  std::optional<Type> RemarkTypeFilter;
  /// Returns a filter object if all the arguments provided are valid regex
  /// types otherwise return an error.
  static Expected<Filters>
  createRemarkFilter(std::optional<FilterMatcher> RemarkNameFilter,
                     std::optional<FilterMatcher> PassNameFilter,
                     std::optional<Type> RemarkTypeFilter) {
    Filters Filter;
    Filter.RemarkNameFilter = std::move(RemarkNameFilter);
    Filter.PassNameFilter = std::move(PassNameFilter);
    Filter.RemarkTypeFilter = RemarkTypeFilter;
    if (auto E = Filter.regexArgumentsValid())
      return E;
    return Filter;
  }
  /// Returns true if the remark satisfies all the provided filters.
  bool filterRemark(const Remark &Remark);

private:
  /// Check if arguments can be parsed as valid regex types.
  Error regexArgumentsValid();
};

// Convert Regex string error to an error object.
inline Error checkRegex(Regex &Regex) {
  std::string Error;
  if (!Regex.isValid(Error))
    return createStringError(make_error_code(std::errc::invalid_argument),
                             Twine("Regex: ", Error));
  return Error::success();
}

/// Abstract counter class used to define the general required methods for
/// counting a remark.
struct Counter {
  GroupBy GroupBy;
  Counter(){};
  Counter(enum GroupBy GroupBy) : GroupBy(GroupBy) {}
  std::optional<std::string> getGroupByKey(const Remark &Remark);

  /// Collect info from remark.
  virtual void collect(const Remark &) = 0;
  /// Output the final count.
  virtual Error print(StringRef OutputFileName) = 0;
  virtual ~Counter() = default;
};

/// Count the remark by looking at the keys provided by the user and the
/// arguments in the remark.
struct KeyCounter : Counter {
  /// The internal object to keep the count for the remarks. The first argument
  /// corresponds to the property we are collecting for this can be either a
  /// source or function. The second argument is a row of integers where each
  /// item in the row is the count for a specified key.
  std::map<std::string, SmallVector<int, 4>> CountByKeysMap;
  /// A set of all the keys found in the remark file. The second argument is the
  /// index of each of those keys which can be used in `CountByKeysMap` to fill
  /// count information for that key.
  MapVector<StringRef, int> KeySetIdxMap;
  KeyCounter(){};
  static Expected<KeyCounter>
  createKeyCounter(enum GroupBy GroupBy, SmallVector<FilterMatcher, 4> &Keys,
                   StringRef Buffer) {
    KeyCounter KC;
    KC.GroupBy = GroupBy;
    for (auto &Key : Keys) {
      if (Key.IsRegex) {
        if (auto E = checkRegex(std::get<Regex>(Key.FilterRE)))
          return E;
      }
    }
    if (auto E = KC.getAllKeysInRemarks(Buffer, Keys))
      return E;
    return KC;
  }
  void collect(const Remark &) override;
  Error print(StringRef OutputFileName) override;

private:
  /// collect all the keys that match the regex descriptions provided and fill
  /// KeySetIdxMap acting as a row for for all the keys that we are interested
  /// in collecting information for.
  Error getAllKeysInRemarks(StringRef Buffer,
                            SmallVector<FilterMatcher, 4> &Keys);
};

struct RemarkCounter : Counter {
  std::map<std::string, int> CountedByRemarksMap;
  RemarkCounter(enum GroupBy GroupBy) : Counter(GroupBy) {}
  void collect(const Remark &) override;
  Error print(StringRef OutputFileName) override;
};
} // namespace remarks

} // namespace llvm
#endif // TOOLS_LLVM_REMARKCOUNTER_H