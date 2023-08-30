//===------------------------- SocketMsgSupport.h -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H
#define LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H

#include "clang/Tooling/ModuleBuildDaemon/Protocol.h"

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

std::string getBufferFromSocketMsg(SocketMsg SocketMsg);
Expected<SocketMsg> getSocketMsgFromBuffer(char *Buffer);
Expected<SocketMsg> readSocketMsgFromSocket(int FD);
llvm::Error writeSocketMsgToSocket(SocketMsg Msg, int FD);
Expected<int> connectAndWriteSocketMsgToSocket(SocketMsg Msg,
                                               StringRef SocketPath);

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

#endif // LLVM_CLANG_TOOLING_MODULEBUILDDAEMON_SOCKETMSGSUPPORT_H