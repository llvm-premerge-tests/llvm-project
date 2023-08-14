//===-- BasicBlockSectionsProfileReader.cpp -------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Implementation of the basic block sections profile reader pass. It parses
// and stores the basic block sections profile file (which is specified via the
// `-basic-block-sections` flag).
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/BasicBlockSectionsProfileReader.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/DebugInfoMetadata.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <llvm/ADT/STLExtras.h>
#include <optional>

using namespace llvm;

char BasicBlockSectionsProfileReader::ID = 0;
INITIALIZE_PASS(BasicBlockSectionsProfileReader, "bbsections-profile-reader",
                "Reads and parses a basic block sections profile.", false,
                false)

bool BasicBlockSectionsProfileReader::isFunctionHot(StringRef FuncName) const {
  return getRawProfileForFunction(FuncName).first;
}

static std::optional<ProfileBBID> ParsePropellerBBID(StringRef str) {
  SmallVector<StringRef, 2> parts;
  str.split(parts, '.');
  unsigned long long BBID;
  if (getAsUnsignedInteger(parts[0], 10, BBID)) {
    return std::nullopt;
  }
  unsigned long long CloneID = 0;
  if (parts.size() > 1 && getAsUnsignedInteger(parts[1], 10, CloneID)) {
    return std::nullopt;
  }
  return ProfileBBID{static_cast<unsigned>(BBID),
                     static_cast<unsigned>(CloneID)};
}

std::pair<bool, RawFunctionProfile>
BasicBlockSectionsProfileReader::getRawProfileForFunction(
    StringRef FuncName) const {
  auto R = RawProgramProfile.find(getAliasName(FuncName));
  return R != RawProgramProfile.end() ? std::pair(true, R->second)
                                      : std::pair(false, RawFunctionProfile());
}

// Basic Block Sections can be enabled for a subset of machine basic blocks.
// This is done by passing a file containing names of functions for which basic
// block sections are desired.  Additionally, machine basic block ids of the
// functions can also be specified for a finer granularity. Moreover, a cluster
// of basic blocks could be assigned to the same section.
// Optionally, a debug-info filename can be specified for each function to allow
// distinguishing internal-linkage functions of the same name.
// A file with basic block sections for all of function main and three blocks
// for function foo (of which 1 and 2 are placed in a cluster) looks like this:
// (Profile for function foo is only loaded when its debug-info filename
// matches 'path/to/foo_file.cc').
// ----------------------------
// list.txt:
// !main
// !foo M=path/to/foo_file.cc
// !!1 2
// !!4
Error BasicBlockSectionsProfileReader::ReadProfile() {
  assert(MBuf);
  line_iterator LineIt(*MBuf, /*SkipBlanks=*/true, /*CommentMarker=*/'#');

  auto invalidProfileError = [&](auto Message) {
    return make_error<StringError>(
        Twine("Invalid profile " + MBuf->getBufferIdentifier() + " at line " +
              Twine(LineIt.line_number()) + ": " + Message),
        inconvertibleErrorCode());
  };

  auto FI = RawProgramProfile.end();

  // Current cluster ID corresponding to this function.
  unsigned CurrentCluster = 0;
  // Current position in the current cluster.
  unsigned CurrentPosition = 0;

  // Temporary set to ensure every basic block ID appears once in the clusters
  // of a function.
  SmallSet<ProfileBBID, 4> FuncBBIDs;

  for (; !LineIt.is_at_eof(); ++LineIt) {
    StringRef S(*LineIt);
    if (S[0] == '@')
      continue;
    // Check for the leading "!"
    if (!S.consume_front("!") || S.empty())
      break;

    if (S.consume_front("!!")) {
      // Skip the profile when we the profile iterator (FI) refers to the
      // past-the-end element.
      if (FI == RawProgramProfile.end())
        continue;
      FI->second.ClonePaths.push_back({});
      SmallVector<StringRef, 5> ClonePath;
      S.split(ClonePath, ' ');

      for (auto BBIDStr : ClonePath) {
        unsigned long long BBID = 0;
        if (getAsUnsignedInteger(BBIDStr, 10, BBID))
          return invalidProfileError(Twine("BB Id expected: '") + BBIDStr +
                                     "'.");
        FI->second.ClonePaths.back().push_back(BBID);
      }
    } else {
      // Check for second "!" which indicates a cluster of basic blocks.
      if (S.consume_front("!")) {
        // Skip the profile when we the profile iterator (FI) refers to the
        // past-the-end element.
        if (FI == RawProgramProfile.end())
          continue;
        SmallVector<StringRef, 4> BBIDs;
        S.split(BBIDs, ' ');
        // Reset current cluster position.
        CurrentPosition = 0;
        for (auto &BBIDStr : BBIDs) {
          auto BBID = ParsePropellerBBID(BBIDStr);
          if (!BBID) {
            return invalidProfileError(Twine("BB Id expected: '") + BBIDStr +
                                       "'.");
          }
          // if (!FuncBBIDs.insert(*BBID).second)
          //   return invalidProfileError(
          //       Twine("Duplicate basic block id found '") + BBIDStr + "'.");

          if (!BBID->BBID && CurrentPosition)
            return invalidProfileError(
                "Entry BB (0) does not begin a cluster.");

          FI->second.RawBBProfiles.emplace_back(
              RawBBProfile{*BBID, CurrentCluster, CurrentPosition++});
        }
        CurrentCluster++;
      } else {
        // This is a function name specifier. It may include a debug info
        // filename specifier starting with `M=`.
        auto [AliasesStr, DIFilenameStr] = S.split(' ');
        SmallString<128> DIFilename;
        if (DIFilenameStr.startswith("M=")) {
          DIFilename =
              sys::path::remove_leading_dotslash(DIFilenameStr.substr(2));
          if (DIFilename.empty())
            return invalidProfileError("Empty module name specifier.");
        } else if (!DIFilenameStr.empty()) {
          return invalidProfileError("Unknown string found: '" + DIFilenameStr +
                                     "'.");
        }
        // Function aliases are separated using '/'. We use the first function
        // name for the cluster info mapping and delegate all other aliases to
        // this one.
        SmallVector<StringRef, 4> Aliases;
        AliasesStr.split(Aliases, '/');
        bool FunctionFound = any_of(Aliases, [&](StringRef Alias) {
          auto It = FunctionNameToDIFilename.find(Alias);
          // No match if this function name is not found in this module.
          if (It == FunctionNameToDIFilename.end())
            return false;
          // Return a match if debug-info-filename is not specified. Otherwise,
          // check for equality.
          return DIFilename.empty() || It->second.equals(DIFilename);
        });
        if (!FunctionFound) {
          // Skip the following profile by setting the profile iterator (FI) to
          // the past-the-end element.
          FI = RawProgramProfile.end();
          continue;
        }
        for (size_t i = 1; i < Aliases.size(); ++i)
          FuncAliasMap.try_emplace(Aliases[i], Aliases.front());

        // Prepare for parsing clusters of this function name.
        // Start a new cluster map for this function name.
        auto R = RawProgramProfile.try_emplace(Aliases.front());
        // Report error when multiple profiles have been specified for the same
        // function.
        if (!R.second)
          return invalidProfileError("Duplicate profile for function '" +
                                     Aliases.front() + "'.");
        FI = R.first;
        CurrentCluster = 0;
        FuncBBIDs.clear();
      }
    }
  }
  return Error::success();
}

bool BasicBlockSectionsProfileReader::doInitialization(Module &M) {
  if (!MBuf)
    return false;
  // Get the function name to debug info filename mapping.
  FunctionNameToDIFilename.clear();
  for (const Function &F : M) {
    SmallString<128> DIFilename;
    if (F.isDeclaration())
      continue;
    DISubprogram *Subprogram = F.getSubprogram();
    if (Subprogram) {
      llvm::DICompileUnit *CU = Subprogram->getUnit();
      if (CU)
        DIFilename = sys::path::remove_leading_dotslash(CU->getFilename());
    }
    [[maybe_unused]] bool inserted =
        FunctionNameToDIFilename.try_emplace(F.getName(), DIFilename).second;
    assert(inserted);
  }
  if (auto Err = ReadProfile())
    report_fatal_error(std::move(Err));
  return false;
}

ImmutablePass *
llvm::createBasicBlockSectionsProfileReaderPass(const MemoryBuffer *Buf) {
  return new BasicBlockSectionsProfileReader(Buf);
}
