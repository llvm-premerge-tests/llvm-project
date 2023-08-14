//===--------------------------- Protocall.cpp ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/Protocall.h"
#include "clang/Basic/Version.h"
#include "clang/Frontend/CompilerInstance.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/BLAKE3.h"

// TODO: Make portable
#if LLVM_ON_UNIX

#include <cerrno>
#include <fstream>
#include <iostream>
#include <signal.h>
#include <spawn.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

using namespace clang;
using namespace llvm;

Expected<int> cc1modbuildd::CreateSocket() {
  int fd;
  if ((fd = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
    std::string msg = "socket create error: " + std::string(strerror(errno));
    return createStringError(inconvertibleErrorCode(), msg);
  }
  return fd;
}

Expected<int> cc1modbuildd::ConnectToSocket(StringRef SocketPath, int FD) {

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(addr));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SocketPath.str().c_str(), sizeof(addr.sun_path) - 1);

  if (connect(FD, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
    close(FD);
    std::string msg = "socket connect error: " + std::string(strerror(errno));
    return createStringError(inconvertibleErrorCode(), msg);
  }

  return FD;
}

bool cc1modbuildd::ModuleBuildDaemonExists(StringRef BasePath) {
  SmallString<128> SocketPath = BasePath;
  SocketPath.append(".sock");

  Expected<int> TestFD = cc1modbuildd::CreateSocket();
  if (!TestFD) {
    consumeError(TestFD.takeError());
    return false;
  }

  Expected<int> ConnectedFD = ConnectToSocket(SocketPath, std::move(*TestFD));
  if (llvm::sys::fs::exists(SocketPath) && ConnectedFD) {
    close(std::move(*ConnectedFD));
    return true;
  }

  consumeError(ConnectedFD.takeError());
  return false;
}

Expected<bool> cc1modbuildd::SpawnModuleBuildDaemon(StringRef BasePath,
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
  return true;
}

SmallString<128> cc1modbuildd::GetBasePath() {
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
  llvm::sys::path::append(BasePath, "clang-" + Key, "mbd");
  return BasePath;
}

llvm::Error cc1modbuildd::ReadMsg(int fd, std::function<int(char*)> func)
{
  char buf[MAX_BUFFER];
  memset(buf, 0, MAX_BUFFER);
  int n = read(fd, buf, MAX_BUFFER);

  if (n < 0) {
    std::string msg = "socket read error: " + std::string(strerror(errno));
    return llvm::make_error<StringError>(msg, inconvertibleErrorCode());
  } else if (n == 0) {
    return llvm::make_error<StringError>("EOF", inconvertibleErrorCode());
  } else {
    func(buf);
  }
  return llvm::Error::success();
}

// Temporary test component used to send messages to cc1modbuildd
int cc1modbuildd::SendMessage(CompilerInstance &Clang, StringRef Argv0,
                              ArrayRef<const char *> Argv, std::string WD,
                              StringRef BasePath) {

  std::vector<std::string> CC1Cmd = Clang.getInvocation().getCC1CommandLine();
  CC1Cmd.insert(CC1Cmd.begin(), Argv0.str());
  cc1modbuildd::Command Command{SCAN, WD, CC1Cmd};

  std::string buffer;
  llvm::raw_string_ostream OS(buffer);
  llvm::yaml::Output yout(OS);

  yout << Command;

  // FIXME: Should not need to append again here
  SmallString<128> SocketPath = BasePath;
  SocketPath.append(".sock");

  Expected<int> MaybeFD = cc1modbuildd::CreateSocket();
  if (!MaybeFD) {
    std::cout << toString(MaybeFD.takeError()) << std::endl;
    return -1;
  }
  consumeError(MaybeFD.takeError());
  int RealFD = std::move(*MaybeFD);

  Expected<int> MaybeWriteFD = cc1modbuildd::ConnectToSocket(SocketPath, RealFD);
  if (!MaybeWriteFD) {
    std::cout << toString(MaybeWriteFD.takeError()) << std::endl;
    return -1;
  }
  int WriteFD = std::move(*MaybeWriteFD);

  // write
  ssize_t message_size = static_cast<ssize_t>(buffer.size());
  if (write(WriteFD, buffer.c_str(), buffer.size()) != message_size) {
    std::perror("write error");
    close(WriteFD);
    return -1;
  }

  char buf[2048];
  memset(buf, 0, 2048);

  // read MAX_BUFFER bytes from client to buf
  int n = read(WriteFD, buf, 2048);

  if (n < 0) {
    perror("read error");
    return -1;
  } else if (n == 0) {
    std::cerr << "EOF" << std::endl;
    return -1;
  } else {
    std::cout << "message: " << buf << std::endl;
  }

  close(WriteFD);
  return 0;
}
#endif // LLVM_ON_UNIX