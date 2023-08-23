//===- Sanitizers.cpp - C Language Family Language Options ----------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the classes from Sanitizers.h
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Sanitizers.h"
#include "clang/Driver/Options.h"
#include "llvm/ADT/Hashing.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/MathExtras.h"
#include "llvm/TargetParser/ARMTargetParser.h"
#include "llvm/TargetParser/Triple.h"
#include <algorithm>

using namespace clang;

// Once LLVM switches to C++17, the constexpr variables can be inline and we
// won't need this.
#define SANITIZER(NAME, ID) constexpr SanitizerMask SanitizerKind::ID;
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  constexpr SanitizerMask SanitizerKind::ID;                                   \
  constexpr SanitizerMask SanitizerKind::ID##Group;
#include "clang/Basic/Sanitizers.def"

SanitizerMask clang::parseSanitizerValue(StringRef Value, bool AllowGroups) {
  SanitizerMask ParsedKind = llvm::StringSwitch<SanitizerMask>(Value)
#define SANITIZER(NAME, ID) .Case(NAME, SanitizerKind::ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  .Case(NAME, AllowGroups ? SanitizerKind::ID##Group : SanitizerMask())
#include "clang/Basic/Sanitizers.def"
    .Default(SanitizerMask());
  return ParsedKind;
}

void clang::serializeSanitizerSet(SanitizerSet Set,
                                  SmallVectorImpl<StringRef> &Values) {
#define SANITIZER(NAME, ID)                                                    \
  if (Set.has(SanitizerKind::ID))                                              \
    Values.push_back(NAME);
#include "clang/Basic/Sanitizers.def"
}

SanitizerMask clang::expandSanitizerGroups(SanitizerMask Kinds) {
#define SANITIZER(NAME, ID)
#define SANITIZER_GROUP(NAME, ID, ALIAS)                                       \
  if (Kinds & SanitizerKind::ID##Group)                                        \
    Kinds |= SanitizerKind::ID;
#include "clang/Basic/Sanitizers.def"
  return Kinds;
}

llvm::hash_code SanitizerMask::hash_value() const {
  return llvm::hash_combine_range(&maskLoToHigh[0], &maskLoToHigh[kNumElem]);
}

namespace clang {
unsigned SanitizerMask::countPopulation() const {
  unsigned total = 0;
  for (const auto &Val : maskLoToHigh)
    total += llvm::popcount(Val);
  return total;
}

llvm::hash_code hash_value(const clang::SanitizerMask &Arg) {
  return Arg.hash_value();
}

StringRef AsanDtorKindToString(llvm::AsanDtorKind kind) {
  switch (kind) {
  case llvm::AsanDtorKind::None:
    return "none";
  case llvm::AsanDtorKind::Global:
    return "global";
  case llvm::AsanDtorKind::Invalid:
    return "invalid";
  }
  return "invalid";
}

llvm::AsanDtorKind AsanDtorKindFromString(StringRef kindStr) {
  return llvm::StringSwitch<llvm::AsanDtorKind>(kindStr)
      .Case("none", llvm::AsanDtorKind::None)
      .Case("global", llvm::AsanDtorKind::Global)
      .Default(llvm::AsanDtorKind::Invalid);
}

StringRef AsanDetectStackUseAfterReturnModeToString(
    llvm::AsanDetectStackUseAfterReturnMode mode) {
  switch (mode) {
  case llvm::AsanDetectStackUseAfterReturnMode::Always:
    return "always";
  case llvm::AsanDetectStackUseAfterReturnMode::Runtime:
    return "runtime";
  case llvm::AsanDetectStackUseAfterReturnMode::Never:
    return "never";
  case llvm::AsanDetectStackUseAfterReturnMode::Invalid:
    return "invalid";
  }
  return "invalid";
}

llvm::AsanDetectStackUseAfterReturnMode
AsanDetectStackUseAfterReturnModeFromString(StringRef modeStr) {
  return llvm::StringSwitch<llvm::AsanDetectStackUseAfterReturnMode>(modeStr)
      .Case("always", llvm::AsanDetectStackUseAfterReturnMode::Always)
      .Case("runtime", llvm::AsanDetectStackUseAfterReturnMode::Runtime)
      .Case("never", llvm::AsanDetectStackUseAfterReturnMode::Never)
      .Default(llvm::AsanDetectStackUseAfterReturnMode::Invalid);
}

bool isExecuteOnlyTarget(const llvm::Triple &Triple,
                         const llvm::opt::ArgList &Args) {
  if (Triple.isPS5())
    return true;
  // On Arm, the clang `-mexecute-only` option is used to generate the
  // execute-only output (no data access to code sections).
  const llvm::opt::Arg *A =
      Args.getLastArg(clang::driver::options::OPT_mexecute_only,
                      clang::driver::options::OPT_mno_execute_only);
  // On Arm, `-target-feature +execute-only` is used to generate the
  // execute-only output from the `clang_cc1` command.
  const std::vector<std::string> Features =
      Args.getAllArgValues(clang::driver::options::OPT_target_feature);

  if ((A &&
       A->getOption().matches(clang::driver::options::OPT_mexecute_only)) ||
      (std::find(Features.begin(), Features.end(), "+execute-only") !=
       Features.end())) {
    // The execute-only output is supported only on ARMv6T2 and ARMv7 and above.
    if (llvm::ARM::parseArchVersion(Triple.getArchName()) > 7 ||
        llvm::ARM::parseArch(Triple.getArchName()) ==
            llvm::ARM::ArchKind::ARMV6T2 ||
        llvm::ARM::parseArch(Triple.getArchName()) ==
            llvm::ARM::ArchKind::ARMV6M)
      return true;
  }

  return false;
}

} // namespace clang
