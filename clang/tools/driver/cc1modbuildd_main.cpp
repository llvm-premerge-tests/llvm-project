//===------- cc1modbuildd_main.cpp - Clang CC1 Module Build Daemon --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/DependencyScanning/DependencyScanningService.h"
#include "clang/Tooling/DependencyScanning/DependencyScanningTool.h"
#include "clang/Tooling/DependencyScanning/ModuleDepCollector.h"
#include "clang/Tooling/ModuleBuildDaemon/Protocall.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

// TODO: Make portable
#if LLVM_ON_UNIX

#include <fstream>
#include <iostream>
#include <signal.h>
#include <sstream>
#include <stdbool.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

#define MAX_BUFFER 2048
#define BASE_FILE "clang-mbd"

static constexpr const char *SocketExtension = ".sock";
static constexpr const char *PidExtension = ".pid";

using namespace llvm;
using namespace clang;

static void Scan(cc1modbuildd::Command Command) {

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
      Command.FullCommandLine, Command.WorkingDirectory, AlreadySeenModules,
      LookupOutput);

  std::cout << toString(MaybeFile.takeError()) << std::endl;

  clang::tooling::dependencies::TranslationUnitDeps TUDeps =
      std::move(*MaybeFile);

  for (auto const &Dep : TUDeps.FileDeps) {
    std::cout << Dep << std::endl;
  }
}

namespace {
class ModuleBuildDaemonServer {
public:
  SmallString<128> BasePath;
  SmallString<128> SocketPath;
  SmallString<128> PidPath;

  ModuleBuildDaemonServer(SmallString<128> Path)
      : BasePath(Path), SocketPath(Path), PidPath(Path) {
    SocketPath.append(SocketExtension);
    PidPath.append(PidExtension);
  }

  ~ModuleBuildDaemonServer() { Shutdown(SIGTERM); }

  int Fork();
  int Launch();
  int Listen();

  // FIXME: Shutdown is not called when computer is powered off
  void Shutdown(int signal) {
    ::unlink(SocketPath.c_str());
    ::shutdown(ListenSocketFD, SHUT_RD);
    ::close(ListenSocketFD);
    exit(EXIT_SUCCESS);
  }

private:
  pid_t Pid = -1;
  int ListenSocketFD = -1;
};

// Required to handle SIGTERM by calling Shutdown
ModuleBuildDaemonServer *DaemonPtr = nullptr;
void HandleSignal(int signal) {
  if (DaemonPtr != nullptr) {
    DaemonPtr->Shutdown(signal);
  }
}
} // namespace

// Forks and detaches process, creating module build daemon
int ModuleBuildDaemonServer::Fork() {

  pid_t pid = ::fork();

  if (pid < 0) {
    exit(EXIT_FAILURE);
  }
  if (pid > 0) {
    exit(EXIT_SUCCESS);
  }

  Pid = ::getpid();

  ::close(STDIN_FILENO);
  ::close(STDOUT_FILENO);
  ::close(STDERR_FILENO);

  freopen("daemon.out", "a", stdout);
  freopen("daemon.err", "a", stderr);

  if (::signal(SIGTERM, HandleSignal) == SIG_ERR)
    // FIXME: should be replaced by error handler (reportError)
    std::cout << "failed to handle SIGTERM" << std::endl;
  if (::signal(SIGHUP, SIG_IGN) == SIG_ERR)
    // FIXME: should be replaced by error handler (reportError)
    std::cout << "failed to ignore SIGHUP" << std::endl;
  if (::setsid() == -1)
    // FIXME: should be replaced by error handler (reportError)
    std::cout << "setsid failed" << std::endl;

  std::cout << "daemon initialization complete!" << std::endl;
  return EXIT_SUCCESS;
}

// Creates unix socket for IPC with module build daemon
int ModuleBuildDaemonServer::Launch() {

  struct sockaddr_un addr;

  // new socket
  if ((ListenSocketFD = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
    std::perror("socket error");
    return -1;
  }

  // set addr to all 0s
  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SocketPath.c_str(), sizeof(addr.sun_path) - 1);

  // bind to local address
  if (bind(ListenSocketFD, (struct sockaddr *)&addr, sizeof(addr)) == -1) {
    std::perror("bind error");
    return -1;
  }

  // set socket to accept incoming connection request
  unsigned MaxBacklog = llvm::hardware_concurrency().compute_thread_count();
  if (::listen(ListenSocketFD, MaxBacklog) == -1) {
    std::perror("listen error");
    return -1;
  }

  return 0;
}

int ModuleBuildDaemonServer::Listen() {

  auto Service = [](int client) {

    auto HandleMsg = [](char* buf) {
      cc1modbuildd::Command Command;
      llvm::yaml::Input yin(buf);

      yin >> Command;
      if (yin.error())
        return -1;

      Scan(Command);
      return 0;
    };

    llvm::Error Err = cc1modbuildd::ReadMsg(client, HandleMsg);
    
    // When a cc1 invokation checks wheather a daemon exists the act of 
    // attempting to make a socket connection on SocketPath triggers the
    // daemon to accept a client connection. Only send response if ReadMsg
    // reads a command sent from the cc1 invocation
    if(Err)
      return std::move(Err);

    std::string msg = "command complete";
    ssize_t message_size = static_cast<ssize_t>(msg.size());
    if (write(client, msg.c_str(), msg.size()) != message_size) {
      close(client);
      std::string msg = "socket write error: " + std::string(strerror(errno));
      return llvm::make_error<StringError>(msg, inconvertibleErrorCode());
    }

    close(client);
  };

  llvm::ThreadPool Pool;
  int client;

  while (true) {

    if ((client = accept(ListenSocketFD, NULL, NULL)) == -1) {
      std::perror("accept error");
      continue;
    }

    std::shared_future<llvm::Error> result = Pool.async(Service, client);
  }
  return 0;
}

int cc1modbuildd_main(ArrayRef<const char *> Argv) {

  SmallString<128> BasePath(Argv[0]);

  StringRef BaseDir = llvm::sys::path::parent_path(BasePath);
  llvm::sys::fs::create_directories(BaseDir);

  ModuleBuildDaemonServer Daemon(BasePath);
  std::cout << Daemon.SocketPath.c_str() << std::endl;

  // Used to handle signals
  DaemonPtr = &Daemon;

  Daemon.Fork();
  Daemon.Launch();
  Daemon.Listen();

  return 0;
}

#endif // LLVM_ON_UNIX