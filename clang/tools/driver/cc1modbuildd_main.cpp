//===------- cc1modbuildd_main.cpp - Clang CC1 Module Build Daemon --------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Tooling/ModuleBuildDaemon/Protocol.h"
#include "clang/Tooling/ModuleBuildDaemon/SocketSupport.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/Threading.h"
#include "llvm/Support/YAMLParser.h"
#include "llvm/Support/YAMLTraits.h"

// TODO: Make portable
#if LLVM_ON_UNIX

#include <errno.h>
#include <fstream>
#include <signal.h>
#include <sstream>
#include <stdbool.h>
#include <string>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <unistd.h>

using namespace llvm;
using namespace clang;

namespace {
class ModuleBuildDaemonServer {
public:
  SmallString<128> BasePath;
  SmallString<128> SocketPath;
  SmallString<128> PidPath;

  ModuleBuildDaemonServer(SmallString<128> Path)
      : BasePath(Path), SocketPath(Path) {
    llvm::sys::path::append(SocketPath, SOCKET_FILE_NAME);
  }

  ~ModuleBuildDaemonServer() { Shutdown(SIGTERM); }

  int Fork();
  int Launch();
  int Listen();
  static llvm::Error Service(int Client);

  void Shutdown(int signal) {
    unlink(SocketPath.c_str());
    shutdown(ListenSocketFD, SHUT_RD);
    close(ListenSocketFD);
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

  pid_t pid = fork();

  if (pid < 0) {
    exit(EXIT_FAILURE);
  }
  if (pid > 0) {
    exit(EXIT_SUCCESS);
  }

  Pid = getpid();

  close(STDIN_FILENO);
  close(STDOUT_FILENO);
  close(STDERR_FILENO);

  SmallString<128> STDOUT = BasePath;
  llvm::sys::path::append(STDOUT, STDOUT_FILE_NAME);
  freopen(STDOUT.c_str(), "a", stdout);

  SmallString<128> STDERR = BasePath;
  llvm::sys::path::append(STDERR, STDERR_FILE_NAME);
  freopen(STDERR.c_str(), "a", stderr);

  if (signal(SIGTERM, HandleSignal) == SIG_ERR) {
    errs() << "failed to handle SIGTERM" << '\n';
    exit(EXIT_FAILURE);
  }
  if (signal(SIGHUP, SIG_IGN) == SIG_ERR) {
    errs() << "failed to ignore SIGHUP" << '\n';
    exit(EXIT_FAILURE);
  }
  if (setsid() == -1) {
    errs() << "setsid failed" << '\n';
    exit(EXIT_FAILURE);
  }

  return EXIT_SUCCESS;
}

// Creates unix socket for IPC with module build daemon
int ModuleBuildDaemonServer::Launch() {

  // new socket
  if ((ListenSocketFD = socket(AF_UNIX, SOCK_STREAM, 0)) == -1) {
    std::perror("Socket create error: ");
    exit(EXIT_FAILURE);
  }

  struct sockaddr_un addr;
  memset(&addr, 0, sizeof(struct sockaddr_un));
  addr.sun_family = AF_UNIX;
  strncpy(addr.sun_path, SocketPath.c_str(), sizeof(addr.sun_path) - 1);

  // bind to local address
  if (bind(ListenSocketFD, (struct sockaddr *)&addr, sizeof(addr)) == -1) {

    // If the socket address is already in use, exit because another module
    // build daemon has successfully launched. When translation units are
    // compiled in parallel, until the socket file is created, all clang
    // invocations will spawn a module build daemon.
    if (errno == EADDRINUSE) {
      close(ListenSocketFD);
      exit(EXIT_SUCCESS);
    }
    std::perror("Socket bind error: ");
    exit(EXIT_FAILURE);
  }

  // set socket to accept incoming connection request
  unsigned MaxBacklog = llvm::hardware_concurrency().compute_thread_count();
  if (listen(ListenSocketFD, MaxBacklog) == -1) {
    std::perror("Socket listen error: ");
    exit(EXIT_FAILURE);
  }

  cc1modbuildd::ub_outs() << "daemon initialization complete!" << '\n';
  return 0;
}

// Function submitted to thread pool with each client connection. Not
// responsible for closing client connections
llvm::Error ModuleBuildDaemonServer::Service(int Client) {

  // Read buffer from client connection
  Expected<std::unique_ptr<char[]>> MaybeClientBuffer =
      cc1modbuildd::readFromSocket(Client);
  if (!MaybeClientBuffer)
    return std::move(MaybeClientBuffer.takeError());
  std::unique_ptr<char[]> ClientBuffer = std::move(*MaybeClientBuffer);

  // Convert buffer into SocketMsg
  Expected<cc1modbuildd::SocketMsg> MaybeClientRequest =
      cc1modbuildd::getSocketMsgFromBuffer(ClientBuffer.get());
  if (!MaybeClientRequest)
    return std::move(MaybeClientRequest.takeError());
  cc1modbuildd::SocketMsg ClientRequest = std::move(*MaybeClientRequest);

  // Handle HANDSHAKE
  // Currently cc1 invocations do not wait for response to HANDSHAKE SocketMsg
  if (ClientRequest.MsgAction == cc1modbuildd::ActionType::HANDSHAKE)
    return llvm::Error::success();

  // Handle REGISTER
  if (ClientRequest.MsgAction == cc1modbuildd::ActionType::REGISTER) {
    llvm::Error ScanErr = cc1modbuildd::scanTranslationUnit(ClientRequest);
    if (ScanErr)
      return std::move(ScanErr);

    std::string ServerResponse = cc1modbuildd::getBufferFromSocketMsg(
        {cc1modbuildd::ActionType::REGISTER,
         cc1modbuildd::StatusType::SUCCESS});
    // Unblocks cc1 invocation
    llvm::Error WriteErr = cc1modbuildd::writeToSocket(ServerResponse, Client);
    if (WriteErr)
      return std::move(WriteErr);

    return llvm::Error::success();
  }

  // Conditional should exist for each ActionType
  llvm_unreachable("Unrecognized Action");
}

int ModuleBuildDaemonServer::Listen() {

  llvm::ThreadPool Pool;
  int Client;

  while (true) {

    if ((Client = accept(ListenSocketFD, NULL, NULL)) == -1) {
      std::perror("Socket accept error: ");
      continue;
    }

    // FIXME: Error messages will be over written as results are returned
    std::shared_future<llvm::Error> result = Pool.async(Service, Client);
    llvm::Error Err = std::move(const_cast<llvm::Error &>(result.get()));

    if (Err) {
      handleAllErrors(std::move(Err), [&](ErrorInfoBase &EIB) {
        errs() << "Error while scanning: " << EIB.message() << '\n';
      });
    }

    close(Client);
  }
  return 0;
}

// Module build daemon is spawned with the following command line:
//
// clang -cc1modbuildd <path>
//
// <path> defines the location of all files created by the module build daemon
// and should follow the format /path/to/dir. For example, `clang -cc1modbuildd
// /tmp/` creates a socket file at `/tmp/mbd.sock`. /tmp is also valid.
//
// When module build daemons are spawned by cc1 invocations, <path> follows the
// format /tmp/clang-<BLAKE3HashOfClangFullVersion>
//
int cc1modbuildd_main(ArrayRef<const char *> Argv) {

  if (Argv.size() < 1) {
    outs() << "spawning a module build daemon requies a command line format of "
              "`clang -cc1modbuildd <path>`. <path> defines where the module "
              "build daemon will create files"
           << '\n';
    return 1;
  }

  // TODO: Add check to confirm BasePath is of correct format
  SmallString<128> BasePath(Argv[0]);
  llvm::sys::fs::create_directories(BasePath);
  ModuleBuildDaemonServer Daemon(BasePath);

  // Used to handle signals
  DaemonPtr = &Daemon;

  Daemon.Fork();
  Daemon.Launch();
  Daemon.Listen();

  return 0;
}

#endif // LLVM_ON_UNIX