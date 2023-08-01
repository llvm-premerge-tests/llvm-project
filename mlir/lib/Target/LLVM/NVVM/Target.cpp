//===- NVVMTarget.h - MLIR LLVM NVVM target compilation ---------*- C++ -*-===//
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
#include "llvm/Support/Path.h"
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

// Register the NVVM dialect, the NVVM translation, the target interface and
// call `LLVMInitializeNVPTX*` if possible.
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
      fileList(targetOptions.getBitcodeFiles()) {

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

#ifdef MLIR_GPU_NVPTX_TARGET_ENABLED
#define DEBUG_TYPE "serialize-to-object"
#include <cuda.h>

static void emitCudaError(const llvm::Twine &expr, const char *buffer,
                          CUresult result, Location loc) {
  const char *error;
  cuGetErrorString(result, &error);
  emitError(loc, expr.concat(" failed with error code ")
                     .concat(llvm::Twine{error})
                     .concat("[")
                     .concat(buffer)
                     .concat("]"));
}

#define RETURN_ON_CUDA_ERROR(expr)                                             \
  do {                                                                         \
    if (auto status = (expr)) {                                                \
      emitCudaError(#expr, jitErrorBuffer, status, loc);                       \
      return {};                                                               \
    }                                                                          \
  } while (false)

namespace {
class SerializeToCubin : public SerializeGPUModuleBase {
public:
  using SerializeGPUModuleBase::SerializeGPUModuleBase;

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule,
                 llvm::TargetMachine &targetMachine) override;
};
} // namespace

std::optional<SmallVector<char, 0>>
SerializeToCubin::moduleToObject(llvm::Module &llvmModule,
                                 llvm::TargetMachine &targetMachine) {
  std::optional<std::string> serializedISA =
      translateToISA(llvmModule, targetMachine);
  if (!serializedISA) {
    getOperation().emitError() << "Failed translating the module to ISA.";
    return std::nullopt;
  }

  LLVM_DEBUG({
    llvm::dbgs() << "ISA for module: "
                 << dyn_cast<gpu::GPUModuleOp>(&getOperation()).getNameAttr()
                 << "\n";
    llvm::dbgs() << *serializedISA << "\n";
    llvm::dbgs().flush();
  });

  auto loc = getOperation().getLoc();
  char jitErrorBuffer[4096] = {0};

  RETURN_ON_CUDA_ERROR(cuInit(0));

  // Linking requires a device context.
  CUdevice device;
  RETURN_ON_CUDA_ERROR(cuDeviceGet(&device, 0));
  CUcontext context;
  RETURN_ON_CUDA_ERROR(cuCtxCreate(&context, 0, device));
  CUlinkState linkState;

  CUjit_option jitOptions[] = {CU_JIT_ERROR_LOG_BUFFER,
                               CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES};
  void *jitOptionsVals[] = {jitErrorBuffer,
                            reinterpret_cast<void *>(sizeof(jitErrorBuffer))};

  RETURN_ON_CUDA_ERROR(cuLinkCreate(2,              /* number of jit options */
                                    jitOptions,     /* jit options */
                                    jitOptionsVals, /* jit option values */
                                    &linkState));

  auto kernelName = dyn_cast<gpu::GPUModuleOp>(getOperation()).getName().str();
  RETURN_ON_CUDA_ERROR(cuLinkAddData(
      linkState, CUjitInputType::CU_JIT_INPUT_PTX,
      const_cast<void *>(static_cast<const void *>(serializedISA->c_str())),
      serializedISA->length(), kernelName.c_str(),
      0,       /* number of jit options */
      nullptr, /* jit options */
      nullptr  /* jit option values */
      ));

  void *cubinData;
  size_t cubinSize;
  RETURN_ON_CUDA_ERROR(cuLinkComplete(linkState, &cubinData, &cubinSize));

  char *cubinAsChar = static_cast<char *>(cubinData);
  auto result = SmallVector<char, 0>(cubinAsChar, cubinAsChar + cubinSize);

  // This will also destroy the cubin data.
  RETURN_ON_CUDA_ERROR(cuLinkDestroy(linkState));
  RETURN_ON_CUDA_ERROR(cuCtxDestroy(context));
  return result;
}
#endif

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
#ifdef MLIR_GPU_NVPTX_TARGET_ENABLED
  // TODO: Replace this serializer for one ending on LLVM or PTX.
  SerializeToCubin serializer(*module, cast<NVVMTargetAttr>(attribute),
                              options);
#else
  // Serialize to LLVM bitcode.
  SerializeGPUModuleBase serializer(*module, cast<NVVMTargetAttr>(attribute),
                                    options);
#endif
  serializer.init();
  return serializer.run();
}
