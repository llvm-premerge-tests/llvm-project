//===---------------------------- Protocol.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/Protocol.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Tooling/ModuleBuildDaemon/SocketMsgSupport.h"
#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/BLAKE3.h"

// TODO: Make portable
#if LLVM_ON_UNIX

#include <cerrno>
#include <filesystem>
#include <fstream>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

using namespace clang;
using namespace llvm;

raw_fd_ostream &cc1modbuildd::unbuff_outs() {
  static raw_fd_ostream S(STDOUT_FILENO, false, true);
  return S;
}

std::string cc1modbuildd::getBasePath() {
  llvm::BLAKE3 Hash;
  Hash.update(getClangFullVersion());
  auto HashResult = Hash.final<sizeof(uint64_t)>();
  uint64_t HashValue =
      llvm::support::endian::read<uint64_t, llvm::support::native>(
          HashResult.data());
  std::string Key = toString(llvm::APInt(64, HashValue), 36, /*Signed*/ false);

  // set paths
  SmallString<128> BasePath;
  llvm::sys::path::system_temp_directory(/*erasedOnReboot*/ true, BasePath);
  llvm::sys::path::append(BasePath, "clang-" + Key);
  return BasePath.c_str();
}

bool cc1modbuildd::daemonExists(StringRef BasePath) {

  SmallString<128> SocketPath = BasePath;
  llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);

  if (!llvm::sys::fs::exists(SocketPath))
    return false;

  Expected<int> ConnectedFD = connectToSocketAndHandshake(SocketPath);
  if (ConnectedFD) {
    close(std::move(*ConnectedFD));
    return true;
  }

  consumeError(ConnectedFD.takeError());
  return false;
}

llvm::Error cc1modbuildd::attemptHandshake(int SocketFD) {

  cc1modbuildd::SocketMsg Request{ActionType::HANDSHAKE, StatusType::REQUEST};
  std::string Buffer = cc1modbuildd::getBufferFromSocketMsg(Request);

  if (llvm::Error Err = writeToSocket(Buffer, SocketFD))
    return std::move(Err);

  Expected<SocketMsg> MaybeServerResponse = readSocketMsgFromSocket(SocketFD);
  if (!MaybeServerResponse)
    return std::move(MaybeServerResponse.takeError());
  SocketMsg ServerResponse = std::move(*MaybeServerResponse);

  assert(ServerResponse.MsgAction == ActionType::HANDSHAKE &&
         "At this point response ActionType should only ever be HANDSHAKE");

  if (ServerResponse.MsgStatus == StatusType::SUCCESS)
    return llvm::Error::success();

  return llvm::make_error<StringError>("Handshake failed",
                                       inconvertibleErrorCode());
}

Expected<int> cc1modbuildd::connectToSocketAndHandshake(StringRef SocketPath) {

  Expected<int> ConnectedFD = connectToSocket(SocketPath);
  if (!ConnectedFD)
    return std::move(ConnectedFD.takeError());

  llvm::Error Err = attemptHandshake(std::move(*ConnectedFD));
  if (Err)
    return std::move(Err);

  return ConnectedFD;
}

llvm::Error cc1modbuildd::spawnModuleBuildDaemon(StringRef BasePath,
                                                 const char *Argv0) {
  std::string BasePathStr = BasePath.str();
  const char *Args[] = {Argv0, "-cc1modbuildd", BasePathStr.c_str(), nullptr};
  pid_t pid;
  int EC = posix_spawn(&pid, Args[0],
                       /*file_actions*/ nullptr,
                       /*spawnattr*/ nullptr, const_cast<char **>(Args),
                       /*envp*/ nullptr);
  if (EC)
    return createStringError(std::error_code(EC, std::generic_category()),
                             "failed to spawn module build daemon process");

  return llvm::Error::success();
}

llvm::Error cc1modbuildd::getModuleBuildDaemon(const char *Argv0,
                                               StringRef BasePath) {

  // If module build daemon already exist return success
  if (cc1modbuildd::daemonExists(BasePath)) {
    return llvm::Error::success();
  }

  if (llvm::Error Err = cc1modbuildd::spawnModuleBuildDaemon(BasePath, Argv0))
    return std::move(Err);

  const unsigned int MICROSEC_IN_SEC = 1000000;
  constexpr unsigned int MAX_TIME = 30 * MICROSEC_IN_SEC;
  const unsigned short INTERVAL = 100;

  unsigned int CumulativeTime = 0;
  unsigned int WaitTime = 0;

  while (CumulativeTime <= MAX_TIME) {
    // Wait a bit then check to see if the module build daemon was created
    usleep(WaitTime);
    if (cc1modbuildd::daemonExists(BasePath))
      return llvm::Error::success();
    CumulativeTime += INTERVAL;
  }

  // After waiting 30 seconds give up
  return llvm::make_error<StringError>(
      "Module build daemon did not exist after spawn attempt",
      inconvertibleErrorCode());
}

Expected<int>
cc1modbuildd::registerTranslationUnit(ArrayRef<const char *> CC1Cmd,
                                      StringRef Argv0, StringRef BasePath,
                                      StringRef CWD) {

  std::vector<std::string> Argv0PlusCC1;
  Argv0PlusCC1.push_back(Argv0.str());
  Argv0PlusCC1.insert(Argv0PlusCC1.end(), CC1Cmd.begin(), CC1Cmd.end());

  // FIXME: Should not need to append again here
  SmallString<128> SocketPath = BasePath;
  llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);

  cc1modbuildd::SocketMsg Request{ActionType::REGISTER, StatusType::REQUEST,
                                  CWD.str(), Argv0PlusCC1};

  Expected<int> MaybeServerFD =
      connectAndWriteSocketMsgToSocket(Request, SocketPath);
  if (!MaybeServerFD)
    return std::move(MaybeServerFD.takeError());

  return std::move(*MaybeServerFD);
}

Expected<std::vector<std::string>> cc1modbuildd::getUpdatedCC1(int ServerFD) {

  // Blocks cc1 invocation until module build daemon is done processing
  // translation unit. Currently receives a SUCCESS message and returns
  // llvm::Error::success() but will eventually recive updated cc1 command line
  Expected<SocketMsg> MaybeServerResponse = readSocketMsgFromSocket(ServerFD);
  if (!MaybeServerResponse)
    return std::move(MaybeServerResponse.takeError());
  SocketMsg ServerResponse = std::move(*MaybeServerResponse);

  // Confirm response is REGISTER and MsgStatus is SUCCESS
  assert(ServerResponse.MsgAction == ActionType::REGISTER &&
         "At this point response ActionType should only ever be REGISTER");

  if (ServerResponse.MsgStatus == StatusType::SUCCESS)
    return ServerResponse.Argv0PlusCC1CommandLine.value();

  return llvm::make_error<StringError>(
      "Daemon failed to processes registered translation unit",
      inconvertibleErrorCode());
}

Expected<std::vector<std::string>>
cc1modbuildd::updateCC1WithModuleBuildDaemon(ArrayRef<const char *> CC1Cmd,
                                             const char *Argv0, StringRef CWD) {

  std::string BasePath = cc1modbuildd::getBasePath();
  std::string ErrMessage;

  // If module build daemon does not exist spawn module build daemon
  llvm::Error DaemonErr = cc1modbuildd::getModuleBuildDaemon(Argv0, BasePath);
  if (DaemonErr) {
    handleAllErrors(std::move(DaemonErr), [&](ErrorInfoBase &EIB) {
      ErrMessage = "Connect to daemon failed: " + EIB.message();
    });
    return llvm::make_error<StringError>(ErrMessage, inconvertibleErrorCode());
  }

  // Send translation unit information to module build daemon for processing
  Expected<int> MaybeServerFD =
      cc1modbuildd::registerTranslationUnit(CC1Cmd, Argv0, BasePath, CWD);
  if (!MaybeServerFD) {
    handleAllErrors(
        std::move(MaybeServerFD.takeError()), [&](ErrorInfoBase &EIB) {
          ErrMessage = "Register translation unit failed: " + EIB.message();
        });
    return llvm::make_error<StringError>(ErrMessage, inconvertibleErrorCode());
  }

  // Wait for response from module build daemon. Response will hopefully be an
  // updated cc1 command line with additional -fmodule-file=<file> flags and
  // implicit module flags removed
  Expected<std::vector<std::string>> MaybeUpdatedCC1 =
      cc1modbuildd::getUpdatedCC1(std::move(*MaybeServerFD));
  if (!MaybeUpdatedCC1) {
    handleAllErrors(std::move(MaybeUpdatedCC1.takeError()),
                    [&](ErrorInfoBase &EIB) {
                      ErrMessage = "Get updated cc1 failed: " + EIB.message();
                    });
    return llvm::make_error<StringError>(ErrMessage, inconvertibleErrorCode());
  }

  // Remove the Argv0 from SocketMsg.Argv0PlusCC1CommandLine
  std::vector<std::string> UpdatedCC1 = std::move(*MaybeUpdatedCC1);
  if (!UpdatedCC1.empty()) {
    UpdatedCC1.erase(UpdatedCC1.begin());
  }
  return UpdatedCC1;
}

#endif // LLVM_ON_UNIX