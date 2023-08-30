//===------------------------ SocketMsgSupport.cpp ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/SocketMsgSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/Protocol.h"
#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"

std::string cc1modbuildd::getBufferFromSocketMsg(SocketMsg SocketMsg) {

  std::string Buffer;
  llvm::raw_string_ostream OS(Buffer);
  llvm::yaml::Output YamlOut(OS);

  YamlOut << SocketMsg;
  return Buffer;
}

Expected<cc1modbuildd::SocketMsg>
cc1modbuildd::getSocketMsgFromBuffer(char *Buffer) {

  SocketMsg ClientRequest;
  llvm::yaml::Input YamlIn(Buffer);
  YamlIn >> ClientRequest;

  if (YamlIn.error()) {
    std::string Msg = "Syntax or semantic error during YAML parsing";
    return llvm::make_error<StringError>(Msg, inconvertibleErrorCode());
  }

  return ClientRequest;
}

Expected<cc1modbuildd::SocketMsg>
cc1modbuildd::readSocketMsgFromSocket(int FD) {
  Expected<std::unique_ptr<char[]>> MaybeResponseBuffer = readFromSocket(FD);
  if (!MaybeResponseBuffer)
    return std::move(MaybeResponseBuffer.takeError());

  // Wait for response from module build daemon
  Expected<SocketMsg> MaybeResponse =
      getSocketMsgFromBuffer(std::move(*MaybeResponseBuffer).get());
  if (!MaybeResponse)
    return std::move(MaybeResponse.takeError());

  return std::move(*MaybeResponse);
}

llvm::Error cc1modbuildd::writeSocketMsgToSocket(SocketMsg Msg, int FD) {

  std::string Buffer = getBufferFromSocketMsg(Msg);
  if (llvm::Error Err = writeToSocket(Buffer, FD))
    return std::move(Err);

  return llvm::Error::success();
}

Expected<int>
cc1modbuildd::connectAndWriteSocketMsgToSocket(SocketMsg Msg,
                                               StringRef SocketPath) {
  Expected<int> MaybeFD = connectToSocket(SocketPath);
  if (!MaybeFD)
    return std::move(MaybeFD.takeError());
  int FD = std::move(*MaybeFD);

  if (llvm::Error Err = writeSocketMsgToSocket(Msg, FD))
    return std::move(Err);

  return FD;
}