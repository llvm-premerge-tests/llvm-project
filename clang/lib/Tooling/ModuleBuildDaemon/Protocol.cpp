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
#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
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

static std::string getWorkingDir() {
  return std::filesystem::current_path().string();
}

raw_fd_ostream &cc1modbuildd::ub_outs() {
  static raw_fd_ostream S(STDOUT_FILENO, false, true);
  return S;
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

SmallString<128> cc1modbuildd::getBasePath() {
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
  return BasePath;
}

llvm::Error cc1modbuildd::attemptHandshake(int SocketFD) {

  cc1modbuildd::SocketMsg Request{ActionType::HANDSHAKE, StatusType::REQUEST};
  std::string Buffer = cc1modbuildd::getBufferFromSocketMsg(Request);

  if (llvm::Error Err = writeToSocket(Buffer, SocketFD))
    return std::move(Err);

  return llvm::Error::success();
}

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

llvm::Error cc1modbuildd::getModuleBuildDaemon(const char *Argv0,
                                               StringRef BasePath) {

  // If module build daemon already exist return success
  if (cc1modbuildd::daemonExists(BasePath)) {
    return llvm::Error::success();
  }

  if (llvm::Error Err = cc1modbuildd::spawnModuleBuildDaemon(BasePath, Argv0))
    return std::move(Err);

  sleep(10);

  // Confirm that module build daemon was created
  if (cc1modbuildd::daemonExists(BasePath))
    return llvm::Error::success();

  return llvm::make_error<StringError>(
      "Module build daemon did not exist after spawn attempt",
      inconvertibleErrorCode());
}

void cc1modbuildd::updateCC1WithModuleBuildDaemon(CompilerInstance &Clang,
                                                  const char *Argv0) {

  SmallString<128> BasePath = cc1modbuildd::getBasePath();

  llvm::Error DaemonErr = cc1modbuildd::getModuleBuildDaemon(Argv0, BasePath);
  if (DaemonErr) {
    handleAllErrors(std::move(DaemonErr), [&](ErrorInfoBase &EIB) {
      errs() << "Connect to daemon failed: " << EIB.message() << "\n";
    });
    return;
  }

  llvm::Error RegisterErr =
      cc1modbuildd::registerTranslationUnit(Clang, Argv0, BasePath);
  if (RegisterErr) {
    handleAllErrors(std::move(RegisterErr), [&](ErrorInfoBase &EIB) {
      errs() << "Register translation unit failed: " << EIB.message() << "\n";
    });
    return;
  }

  return;
}

llvm::Error cc1modbuildd::registerTranslationUnit(CompilerInstance &Clang,
                                                  StringRef Argv0,
                                                  StringRef BasePath) {

  std::vector<std::string> CC1Cmd = Clang.getInvocation().getCC1CommandLine();
  CC1Cmd.insert(CC1Cmd.begin(), Argv0.str());
  cc1modbuildd::SocketMsg Request{ActionType::REGISTER, StatusType::REQUEST,
                                  getWorkingDir(), CC1Cmd};

  std::string Buffer = getBufferFromSocketMsg(Request);

  // FIXME: Should not need to append again here
  SmallString<128> SocketPath = BasePath;
  llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);

  Expected<int> MaybeServerFD = connectAndWriteToSocket(Buffer, SocketPath);
  if (!MaybeServerFD)
    return std::move(MaybeServerFD.takeError());

  // Blocks cc1 invocation until module build daemon is done processing
  // translation unit. Currently receives a SUCCESS message and returns
  // llvm::Error::success() but will eventually recive updated cc1 command line
  Expected<std::unique_ptr<char[]>> MaybeResponseBuffer =
      readFromSocket(std::move(*MaybeServerFD));
  if (!MaybeResponseBuffer)
    return std::move(MaybeResponseBuffer.takeError());

  // Wait for response from module build daemon
  Expected<SocketMsg> MaybeResponse =
      getSocketMsgFromBuffer(std::move(*MaybeResponseBuffer).get());
  if (!MaybeResponse)
    return std::move(MaybeResponse.takeError());
  SocketMsg ServerResponse = std::move(*MaybeResponse);

  assert(ServerResponse.MsgAction == ActionType::REGISTER &&
         "ActionType should only be REGISTER");
  if (ServerResponse.MsgStatus == StatusType::SUCCESS)
    return llvm::Error::success();

  return llvm::make_error<StringError>(
      "Daemon failed to processes registered translation unit",
      inconvertibleErrorCode());
}

llvm::Error cc1modbuildd::scanTranslationUnit(SocketMsg Request) {

  tooling::dependencies::DependencyScanningService Service(
      tooling::dependencies::ScanningMode::DependencyDirectivesScan,
      tooling::dependencies::ScanningOutputFormat::Full,
      /*OptimizeArgs*/ false,
      /*EagerLoadModules*/ false);

  tooling::dependencies::DependencyScanningTool Tool(Service);

  llvm::DenseSet<clang::tooling::dependencies::ModuleID> AlreadySeenModules;
  auto LookupOutput = [&](const tooling::dependencies::ModuleID &MID,
                          tooling::dependencies::ModuleOutputKind MOK) {
    return MID.ContextHash;
  };

  auto MaybeFile = Tool.getTranslationUnitDependencies(
      Request.Argv0PlusCC1CommandLine.value(), Request.WorkingDirectory.value(),
      AlreadySeenModules, LookupOutput);

  if (!MaybeFile)
    return std::move(MaybeFile.takeError());

  tooling::dependencies::TranslationUnitDeps TUDeps = std::move(*MaybeFile);

  // For now write dependencies to log file
  for (auto const &Dep : TUDeps.FileDeps) {
    cc1modbuildd::ub_outs() << Dep << '\n';
  }

  if (!TUDeps.ModuleGraph.empty())
    errs() << "Warning: translation unit contained modules. Module build "
              "daemon not yet able to build modules"
           << '\n';

  return llvm::Error::success();
}

#endif // LLVM_ON_UNIX