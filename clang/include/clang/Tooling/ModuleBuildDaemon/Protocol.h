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
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

#define MAX_BUFFER 4096
#define SOCKET_FILE_NAME "mbd.sock"
#define STDOUT_FILE_NAME "mbd.out"
#define STDERR_FILE_NAME "mbd.err"

using namespace clang;
using namespace llvm;

namespace cc1modbuildd {

enum class ActionType { REGISTER, HANDSHAKE };
enum class StatusType { REQUEST, SUCCESS, FAILURE };

struct SocketMsg {
  ActionType MsgAction;
  StatusType MsgStatus;
  std::optional<std::string> WorkingDirectory;
  // First element needs to be path to compiler
  std::optional<std::vector<std::string>> Argv0PlusCC1CommandLine;

  SocketMsg() = default;

  SocketMsg(ActionType Action, StatusType Status,
            const std::optional<std::string> &CurrentWD,
            const std::optional<std::vector<std::string>> &CommandLine)
      : MsgAction(Action), MsgStatus(Status), WorkingDirectory(CurrentWD),
        Argv0PlusCC1CommandLine(CommandLine) {}

  SocketMsg(ActionType Action, StatusType Status)
      : MsgAction(Action), MsgStatus(Status), WorkingDirectory(std::nullopt),
        Argv0PlusCC1CommandLine(std::nullopt) {}
};

// Create unbuffered STDOUT stream so that any logging done by module build
// daemon can be viewed without having to terminate the process
raw_fd_ostream &ub_outs();

SmallString<128> getBasePath();

bool daemonExists(StringRef BasePath);

std::string getBufferFromSocketMsg(SocketMsg Command);

Expected<SocketMsg> getSocketMsgFromBuffer(char *Buffer);

Expected<int> connectToSocketAndHandshake(StringRef SocketPath);

llvm::Error attemptHandshake(int SocketFD);

llvm::Error getModuleBuildDaemon(const char *Argv0, StringRef BasePath);

llvm::Error spawnModuleBuildDaemon(StringRef BasePath, const char *Argv0);

llvm::Error registerTranslationUnit(CompilerInstance &Clang, StringRef Argv0,
                                    StringRef BasePath);

// Work in progress. Eventually function will modify CC1 command line to include
// path to modules already built by the daemon
void updateCC1WithModuleBuildDaemon(CompilerInstance &Clang, const char *Argv0);

llvm::Error scanTranslationUnit(SocketMsg Command);

} // namespace cc1modbuildd

template <>
struct llvm::yaml::ScalarEnumerationTraits<cc1modbuildd::StatusType> {
  static void enumeration(IO &io, cc1modbuildd::StatusType &value) {
    io.enumCase(value, "REQUEST", cc1modbuildd::StatusType::REQUEST);
    io.enumCase(value, "SUCCESS", cc1modbuildd::StatusType::SUCCESS);
    io.enumCase(value, "FAILURE", cc1modbuildd::StatusType::FAILURE);
  }
};

template <>
struct llvm::yaml::ScalarEnumerationTraits<cc1modbuildd::ActionType> {
  static void enumeration(IO &io, cc1modbuildd::ActionType &value) {
    io.enumCase(value, "REGISTER", cc1modbuildd::ActionType::REGISTER);
    io.enumCase(value, "HANDSHAKE", cc1modbuildd::ActionType::HANDSHAKE);
  }
};

template <> struct llvm::yaml::MappingTraits<cc1modbuildd::SocketMsg> {
  static void mapping(IO &io, cc1modbuildd::SocketMsg &info) {
    io.mapRequired("Action", info.MsgAction);
    io.mapRequired("Status", info.MsgStatus);
    io.mapOptional("WorkingDirectory", info.WorkingDirectory);
    io.mapOptional("FullCommandLine", info.Argv0PlusCC1CommandLine);
  }
};

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_PROTOCAL_H
