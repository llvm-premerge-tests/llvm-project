//===- Target.cpp - MLIR LLVM NVVM target compilation -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines NVVM target related functions including registration
// calls for the `#nvvm.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/NVVM/Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Target/LLVM/NVVM/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/NVVM/NVVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/Process.h"
#include "llvm/Support/Program.h"
#include "llvm/Support/TargetSelect.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::NVVM;

#ifndef __DEFAULT_CUDATOOLKIT_PATH__
#define __DEFAULT_CUDATOOLKIT_PATH__ ""
#endif

namespace {
// Implementation of the `TargetAttrInterface` model.
class NVVMTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<NVVMTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;
};
} // namespace

// Register the NVVM dialect, the NVVM translation & the target interface.
void mlir::registerNVVMTarget(DialectRegistry &registry) {
  registerNVVMDialectTranslation(registry);
  registry.addExtension(+[](MLIRContext *ctx, NVVM::NVVMDialect *dialect) {
    NVVMTargetAttr::attachInterface<NVVMTargetAttrImpl>(*ctx);
  });
}

void mlir::registerNVVMTarget(MLIRContext &context) {
  DialectRegistry registry;
  registerNVVMTarget(registry);
  context.appendDialectRegistry(registry);
}

// Search for the CUDA toolkit path.
StringRef mlir::NVVM::getCUDAToolkitPath() {
  if (const char *var = std::getenv("CUDA_ROOT"))
    return var;
  if (const char *var = std::getenv("CUDA_HOME"))
    return var;
  if (const char *var = std::getenv("CUDA_PATH"))
    return var;
  return __DEFAULT_CUDATOOLKIT_PATH__;
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, NVVMTargetAttr target,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, target.getTriple(), target.getChip(),
                     target.getFeatures(), target.getO()),
      target(target), toolkitPath(targetOptions.getToolkitPath()),
      fileList(targetOptions.getLinkFiles()) {

  // If `targetOptions` have an empty toolkitPath use `getCUDAToolkitPath`
  if (toolkitPath.empty())
    toolkitPath = getCUDAToolkitPath();

  // Append the files in the target attribute.
  if (ArrayAttr files = target.getLink())
    for (Attribute attr : files.getValue())
      if (auto file = dyn_cast<StringAttr>(attr))
        fileList.push_back(file.str());

  // Append libdevice to the files to be loaded.
  (void)appendStandardLibs();
}

void SerializeGPUModuleBase::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
  // If the `NVPTX` LLVM target was built, initialize it.
#if MLIR_CUDA_CONVERSIONS_ENABLED == 1
    LLVMInitializeNVPTXTarget();
    LLVMInitializeNVPTXTargetInfo();
    LLVMInitializeNVPTXTargetMC();
    LLVMInitializeNVPTXAsmPrinter();
#endif
  });
}

NVVMTargetAttr SerializeGPUModuleBase::getTarget() const { return target; }

StringRef SerializeGPUModuleBase::getToolkitPath() const { return toolkitPath; }

ArrayRef<std::string> SerializeGPUModuleBase::getFileList() const {
  return fileList;
}

// Try to append `libdevice` from a CUDA toolkit installation.
LogicalResult SerializeGPUModuleBase::appendStandardLibs() {
  StringRef pathRef = getToolkitPath();
  if (pathRef.size()) {
    SmallVector<char, 256> path;
    path.insert(path.begin(), pathRef.begin(), pathRef.end());
    pathRef = StringRef(path.data(), path.size());
    if (!llvm::sys::fs::is_directory(pathRef)) {
      getOperation().emitError() << "CUDA path: " << pathRef
                                 << " does not exist or is not a directory.\n";
      return failure();
    }
    llvm::sys::path::append(path, "nvvm", "libdevice", "libdevice.10.bc");
    pathRef = StringRef(path.data(), path.size());
    if (!llvm::sys::fs::is_regular_file(pathRef)) {
      getOperation().emitError() << "LibDevice path: " << pathRef
                                 << " does not exist or is not a file.\n";
      return failure();
    }
    fileList.push_back(pathRef.str());
  }
  return success();
}

std::optional<SmallVector<std::unique_ptr<llvm::Module>>>
SerializeGPUModuleBase::loadBitcodeFiles(llvm::Module &module,
                                         llvm::TargetMachine &targetMachine) {
  SmallVector<std::unique_ptr<llvm::Module>> bcFiles;
  if (failed(loadBitcodeFilesFromList(module.getContext(), targetMachine,
                                      fileList, bcFiles, true)))
    return std::nullopt;
  return bcFiles;
}

#if MLIR_CUDA_CONVERSIONS_ENABLED == 1
namespace {
class NVPTXSerializer : public SerializeGPUModuleBase {
public:
  NVPTXSerializer(Operation &module, NVVMTargetAttr target,
                  const gpu::TargetOptions &targetOptions);

  // Compile PTX to cubin.
  std::optional<SmallVector<char, 0>>
  compileToBinary(const std::string &ptxCode);

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule,
                 llvm::TargetMachine &targetMachine) override;

private:
  using TmpFile = std::pair<llvm::SmallString<128>, llvm::FileRemover>;

  // Create a temp file.
  std::optional<TmpFile> createTemp(StringRef name, StringRef suffix);

  // Find the PTXAS compiler. The search order is:
  // 1. The toolkit path in `targetOptions`.
  // 2. In the system PATH.
  // 3. The path from `getCUDAToolkitPath()`.
  std::optional<std::string> findPtxas() const;

  // Target options.
  gpu::TargetOptions targetOptions;
};
} // namespace

NVPTXSerializer::NVPTXSerializer(Operation &module, NVVMTargetAttr target,
                                 const gpu::TargetOptions &targetOptions)
    : SerializeGPUModuleBase(module, target, targetOptions),
      targetOptions(targetOptions) {}

std::optional<NVPTXSerializer::TmpFile>
NVPTXSerializer::createTemp(StringRef name, StringRef suffix) {
  llvm::SmallString<128> filename;
  std::error_code ec =
      llvm::sys::fs::createTemporaryFile(name, suffix, filename);
  if (ec) {
    getOperation().emitError() << "Couldn't create the temp file: `" << filename
                               << "`, error message: " << ec.message();
    return std::nullopt;
  }
  return TmpFile(filename, llvm::FileRemover(filename.c_str()));
}

std::optional<std::string> NVPTXSerializer::findPtxas() const {
  // Find the `ptxas` compiler.
  // 1. Check the toolkit path given in the command line.
  StringRef pathRef = targetOptions.getToolkitPath();
  SmallVector<char, 256> path;
  if (pathRef.size()) {
    path.insert(path.begin(), pathRef.begin(), pathRef.end());
    llvm::sys::path::append(path, "bin", "ptxas");
    if (llvm::sys::fs::can_execute(path))
      return StringRef(path.data(), path.size()).str();
  }

  // 2. Check PATH.
  if (std::optional<std::string> ptxasCompiler =
          llvm::sys::Process::FindInEnvPath("PATH", "ptxas"))
    return *ptxasCompiler;

  // 3. Check `getCUDAToolkitPath()`.
  pathRef = getCUDAToolkitPath();
  path.clear();
  if (pathRef.size()) {
    path.insert(path.begin(), pathRef.begin(), pathRef.end());
    llvm::sys::path::append(path, "bin", "ptxas");
    if (llvm::sys::fs::can_execute(path))
      return StringRef(path.data(), path.size()).str();
  }
  return std::nullopt;
}

// TODO: clean this method & have a generic tool driver, switch to the
// `nvptxcompiler` library or never emit binaries with this mechanism and let
// another stage take care of it.
std::optional<SmallVector<char, 0>>
NVPTXSerializer::compileToBinary(const std::string &ptxCode) {
  // Find the PTXAS compiler.
  std::optional<std::string> ptxasCompiler = findPtxas();
  if (!ptxasCompiler) {
    getOperation().emitError()
        << "Couldn't find the `ptxas` compiler. Please specify the toolkit "
           "path, add the compiler to $PATH, or set one of the environment "
           "variables in `NVVM::getCUDAToolkitPath()`.";
    return std::nullopt;
  }

  // Base name for all temp files: mlir-<module name>-<target triple>-<chip>.
  std::string basename = llvm::formatv(
      "mlir-{0}-{1}-{2}",
      dyn_cast<gpu::GPUModuleOp>(&getOperation()).getNameAttr().getValue(),
      getTarget().getTriple(), getTarget().getChip());

  // Create temp files:
  std::optional<TmpFile> ptxFile = createTemp(basename, "%%.ptx");
  if (!ptxFile)
    return std::nullopt;
  std::optional<TmpFile> stdoutFile = createTemp(basename, "%%.stdout");
  if (!stdoutFile)
    return std::nullopt;
  std::optional<TmpFile> stderrFile = createTemp(basename, "%%.stderr");
  if (!stderrFile)
    return std::nullopt;
  std::optional<TmpFile> cubinFile = createTemp(basename, "%%.cubin");
  if (!cubinFile)
    return std::nullopt;

  std::error_code ec;
  // Dump the PTX to a temp file.
  {
    llvm::raw_fd_ostream ptxStream(ptxFile->first, ec);
    if (ec) {
      getOperation().emitError()
          << "Couldn't open the file: `" << ptxFile->first
          << "`, error message: " << ec.message();
      return std::nullopt;
    }
    ptxStream << ptxCode;
    if (ptxStream.has_error()) {
      getOperation().emitError()
          << "An error occurred while writing the PTX to: `" << ptxFile->first
          << "`.";
      return std::nullopt;
    }
    ptxStream.flush();
  }

  // Create PTX args.
  SmallVector<StringRef, 12> ptxasArgs(
      {StringRef("ptxas"), StringRef("-arch"), getTarget().getChip(),
       StringRef(ptxFile->first), StringRef("-o"),
       StringRef(cubinFile->first)});

  std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>> cmdOpts =
      targetOptions.tokenizeCmdOptions();
  for (auto arg : cmdOpts.second)
    ptxasArgs.push_back(arg);

  std::optional<StringRef> redirects[] = {
      std::nullopt,
      stdoutFile->first,
      stderrFile->first,
  };

  // Invoke PTXAS.
  std::string message;
  if (llvm::sys::ExecuteAndWait(ptxasCompiler.value(), ptxasArgs,
                                /*Env=*/std::nullopt,
                                /*Redirects=*/redirects,
                                /*SecondsToWait=*/0,
                                /*MemoryLimit=*/0,
                                /*ErrMsg=*/&message)) {
    if (message.empty()) {
      llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ptxasStderr =
          llvm::MemoryBuffer::getFile(stderrFile->first);
      if (ptxasStderr)
        getOperation().emitError()
            << "PTXAS invocation failed. PTXAS error log:\n"
            << ptxasStderr->get()->getBuffer();
      else
        getOperation().emitError() << "PTXAS invocation failed.";
      return std::nullopt;
    }
    getOperation().emitError()
        << "PTXAS invocation failed, error message: " << message;
    return std::nullopt;
  }

// Dump the output of PTXAS, helpful if the verbose flag was passed.
#define DEBUG_TYPE "serialize-to-binary"
  LLVM_DEBUG({
    llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> ptxasStdout =
        llvm::MemoryBuffer::getFile(stdoutFile->first);
    if (ptxasStdout && (*ptxasStdout)->getBuffer().size()) {
      llvm::dbgs() << "PTXAS output for module: "
                   << dyn_cast<gpu::GPUModuleOp>(&getOperation()).getNameAttr()
                   << "\n";
      llvm::dbgs() << (*ptxasStdout)->getBuffer() << "\n";
      llvm::dbgs().flush();
    }
  });
#undef DEBUG_TYPE

  // Read the cubin file.
  llvm::ErrorOr<std::unique_ptr<llvm::MemoryBuffer>> cubinBuffer =
      llvm::MemoryBuffer::getFile(cubinFile->first);
  if (!cubinBuffer) {
    getOperation().emitError()
        << "Couldn't open the file: `" << cubinFile->first
        << "`, error message: " << cubinBuffer.getError().message();
    return std::nullopt;
  }
  StringRef cubinStr = (*cubinBuffer)->getBuffer();
  return SmallVector<char, 0>(cubinStr.begin(), cubinStr.end());
}

std::optional<SmallVector<char, 0>>
NVPTXSerializer::moduleToObject(llvm::Module &llvmModule,
                                llvm::TargetMachine &targetMachine) {
  // Return LLVM IR if the compilation target is offload.
#define DEBUG_TYPE "serialize-to-llvm"
  LLVM_DEBUG({
    llvm::dbgs() << "LLVM IR for module: "
                 << dyn_cast<gpu::GPUModuleOp>(&getOperation()).getNameAttr()
                 << "\n";
    llvm::dbgs() << llvmModule << "\n";
    llvm::dbgs().flush();
  });
#undef DEBUG_TYPE
  if (targetOptions.getCompilationTarget() == gpu::TargetOptions::offload)
    return SerializeGPUModuleBase::moduleToObject(llvmModule, targetMachine);

  // Emit PTX code.
  std::optional<std::string> serializedISA =
      translateToISA(llvmModule, targetMachine);
  if (!serializedISA) {
    getOperation().emitError() << "Failed translating the module to ISA.";
    return std::nullopt;
  }
#define DEBUG_TYPE "serialize-to-ptx"
  LLVM_DEBUG({
    llvm::dbgs() << "PTX for module: "
                 << dyn_cast<gpu::GPUModuleOp>(&getOperation()).getNameAttr()
                 << "\n";
    llvm::dbgs() << *serializedISA << "\n";
    llvm::dbgs().flush();
  });
#undef DEBUG_TYPE

  // Return PTX if the compilation target is assembly.
  if (targetOptions.getCompilationTarget() == gpu::TargetOptions::assembly)
    return SmallVector<char, 0>(serializedISA->begin(), serializedISA->end());

  // Compile to binary.
  return compileToBinary(*serializedISA);
}
#endif // MLIR_CUDA_CONVERSIONS_ENABLED == 1

std::optional<SmallVector<char, 0>>
NVVMTargetAttrImpl::serializeToObject(Attribute attribute, Operation *module,
                                      const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
#if MLIR_CUDA_CONVERSIONS_ENABLED == 1
  NVPTXSerializer serializer(*module, cast<NVVMTargetAttr>(attribute), options);
  serializer.init();
  return serializer.run();
#else
  module->emitError(
      "The `NVPTX` target was not built. Please enable it when building LLVM.");
  return std::nullopt;
#endif // MLIR_CUDA_CONVERSIONS_ENABLED == 1
}
