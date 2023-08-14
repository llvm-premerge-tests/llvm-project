//===---------------------------- Protocall.h -----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_PROTOCAL_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_PROTOCAL_H

#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

#define MAX_BUFFER 2048

using namespace clang;
using namespace llvm;

namespace cc1modbuildd {

enum Action { SCAN };

struct Command {
  Action CMD;
  std::string WorkingDirectory;
  std::vector<std::string> FullCommandLine;
};

Expected<int> CreateSocket();

Expected<int> ConnectToSocket(StringRef SocketPath, int FD);

bool ModuleBuildDaemonExists(StringRef BasePath);

Expected<bool> SpawnModuleBuildDaemon(StringRef BasePath, const char *Argv0);

SmallString<128> GetBasePath();

llvm::Error ReadMsg(int fd, std::function<int(char*)> func);

int SendMessage(CompilerInstance &Clang, StringRef Argv0,
                ArrayRef<const char *> Argv, std::string WD,
                StringRef BasePath);

} // namespace cc1modbuildd

template <> struct llvm::yaml::ScalarEnumerationTraits<cc1modbuildd::Action> {
  static void enumeration(IO &io, cc1modbuildd::Action &value) {
    io.enumCase(value, "SCAN", cc1modbuildd::SCAN);
  }
};

template <> struct llvm::yaml::MappingTraits<cc1modbuildd::Command> {
  static void mapping(IO &io, cc1modbuildd::Command &info) {
    io.mapRequired("CMD", info.CMD);
    io.mapRequired("WorkingDirectory", info.WorkingDirectory);
    io.mapRequired("FullCommandLine", info.FullCommandLine);
  }
};

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_PROTOCAL_H
