//===- unittests/Interpreter/CudaTest.cpp --- Interactive CUDA tests ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Unit tests for interactive CUDA in Clang interpreter
//
//===----------------------------------------------------------------------===//

#include "clang/Basic/Version.h"
#include "clang/Config/config.h"

#include "clang/Frontend/CompilerInstance.h"
#include "clang/Interpreter/Interpreter.h"

#include "llvm/Support/TargetSelect.h"

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace {

std::string MakeResourcesPath() {
  using namespace llvm;
#ifdef LLVM_BINARY_DIR
  StringRef Dir = LLVM_BINARY_DIR;
#else
  // Dir is bin/ or lib/, depending on where BinaryPath is.
  void *MainAddr = (void *)(intptr_t)MakeResourcesPath;
  std::string BinaryPath =
      llvm::sys::fs::getMainExecutable(/*Argv0=*/nullptr, MainAddr);

  // build/tools/clang/unittests/Interpreter/Executable -> build/
  StringRef Dir = sys::path::parent_path(BinaryPath);

  Dir = sys::path::parent_path(Dir);
  Dir = sys::path::parent_path(Dir);
  Dir = sys::path::parent_path(Dir);
  Dir = sys::path::parent_path(Dir);
#endif // LLVM_BINARY_DIR
  SmallString<128> P(Dir);
  sys::path::append(P, CLANG_INSTALL_LIBDIR_BASENAME, "clang",
                    CLANG_VERSION_MAJOR_STRING);
  return P.str().str();
}

static std::unique_ptr<clang::Interpreter>
createInterpreter(const std::vector<const char *> &ExtraArgs = {}) {
  static bool firstrun = true;
  if (firstrun) {
    llvm::InitializeAllTargetInfos();
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllAsmPrinters();

    firstrun = false;
  }

  clang::IncrementalCompilerBuilder CB;

  // Help find cuda's runtime headers.
  std::string ResourceDir = MakeResourcesPath();

  std::vector Args = {"-resource-dir", ResourceDir.c_str(), "-std=c++20"};
  Args.insert(Args.end(), ExtraArgs.begin(), ExtraArgs.end());
  CB.SetCompilerArgs(Args);

  // Create the device code compiler
  std::unique_ptr<clang::CompilerInstance> DeviceCI;
  CB.SetOffloadArch("sm_35");
  DeviceCI = cantFail(CB.CreateCudaDevice());

  std::unique_ptr<clang::CompilerInstance> CI;
  CI = cantFail(CB.CreateCudaHost());

  auto Interp = cantFail(
      clang::Interpreter::createWithCUDA(std::move(CI), std::move(DeviceCI)));

  return Interp;
}

enum {
  // Defined in CUDA Runtime API
  cudaErrorNoDevice = 100,
};

TEST(InteractiveCudaTest, Sanity) {
  std::unique_ptr<clang::Interpreter> Interp = createInterpreter();

#ifdef LIBCUDART_PATH
  auto Err = Interp->LoadDynamicLibrary(LIBCUDART_PATH);
  if (Err) { // CUDA runtime is not usable, cannot continue testing
    consumeError(std::move(Err));
    return;
  }
#else
  return;
#endif

  // Check if we have any GPU for test
  int CudaError = 0;
  auto GpuCheckCommand = std::string(R"(
    int device_id = -1;
    int *error = (int *))" + std::to_string((uintptr_t)&CudaError) +
                                     R"(;
    *error = cudaGetDevice(&device_id);
  )");
  cantFail(Interp->ParseAndExecute(GpuCheckCommand));
  if (CudaError == cudaErrorNoDevice) {
    // No GPU is available on this machine, cannot continue testing
    return;
  }
  ASSERT_EQ(CudaError, 0);

  int HostSum = 0;
  auto Command1 = std::string(R"(
    __host__ __device__ inline int sum(int a, int b){ return a + b; }
    __global__ void kernel(int * output){ *output = sum(40,2); }
    int *hostsum = (int *) )") +
                  std::to_string((uintptr_t)&HostSum) +
                  R"(;
    *hostsum = sum(41,1);)";
  cantFail(Interp->ParseAndExecute(Command1));

  int DeviceSum = 0;
  auto Command2 = std::string(R"(
    int *devicesum = (int *))" +
                              std::to_string((uintptr_t)&DeviceSum) +
                              R"(;
    int *deviceVar;
    *error |= cudaMalloc((void **) &deviceVar, sizeof(int));
    kernel<<<1,1>>>(deviceVar);
    *error |= cudaGetLastError();
    *error |= cudaMemcpy(devicesum, deviceVar, sizeof(int), cudaMemcpyDeviceToHost);
    *error |= cudaGetLastError();
)");
  cantFail(Interp->ParseAndExecute(Command2));

  ASSERT_EQ(HostSum, 42);
  ASSERT_EQ(DeviceSum, 42);
  ASSERT_EQ(CudaError, 0);
}

} // end anonymous namespace
