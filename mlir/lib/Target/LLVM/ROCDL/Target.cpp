//===- Target.cpp - MLIR LLVM ROCDL target compilation ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines ROCDL target related functions including registration
// calls for the `#rocdl.target` compilation attribute.
//
//===----------------------------------------------------------------------===//

#include "mlir/Target/LLVM/ROCDL/Target.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Target/LLVM/ROCDL/Utils.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/ROCDL/ROCDLToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Export.h"

#include "llvm/MC/MCAsmBackend.h"
#include "llvm/MC/MCAsmInfo.h"
#include "llvm/MC/MCCodeEmitter.h"
#include "llvm/MC/MCContext.h"
#include "llvm/MC/MCInstrInfo.h"
#include "llvm/MC/MCObjectFileInfo.h"
#include "llvm/MC/MCObjectWriter.h"
#include "llvm/MC/MCParser/MCTargetAsmParser.h"
#include "llvm/MC/MCRegisterInfo.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/MC/MCSubtargetInfo.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/TargetParser/TargetParser.h"

#include <cstdlib>

using namespace mlir;
using namespace mlir::ROCDL;

#ifndef __DEFAULT_ROCM_PATH__
#define __DEFAULT_ROCM_PATH__ ""
#endif

namespace {
// Implementation of the `TargetAttrInterface` model.
class ROCDLTargetAttrImpl
    : public gpu::TargetAttrInterface::FallbackModel<ROCDLTargetAttrImpl> {
public:
  std::optional<SmallVector<char, 0>>
  serializeToObject(Attribute attribute, Operation *module,
                    const gpu::TargetOptions &options) const;
};
} // namespace

// Register the ROCDL dialect, the ROCDL translation and the target interface.
void mlir::registerROCDLTarget(DialectRegistry &registry) {
  registerROCDLDialectTranslation(registry);
  registry.addExtension(+[](MLIRContext *ctx, ROCDL::ROCDLDialect *dialect) {
    ROCDLTargetAttr::attachInterface<ROCDLTargetAttrImpl>(*ctx);
  });
}

void mlir::registerROCDLTarget(MLIRContext &context) {
  DialectRegistry registry;
  registerROCDLTarget(registry);
  context.appendDialectRegistry(registry);
}

// Search for the ROCM path.
StringRef mlir::ROCDL::getROCMPath() {
  if (const char *var = std::getenv("ROCM_PATH"))
    return var;
  if (const char *var = std::getenv("ROCM_ROOT"))
    return var;
  if (const char *var = std::getenv("ROCM_HOME"))
    return var;
  return __DEFAULT_ROCM_PATH__;
}

SerializeGPUModuleBase::SerializeGPUModuleBase(
    Operation &module, ROCDLTargetAttr target,
    const gpu::TargetOptions &targetOptions)
    : ModuleToObject(module, target.getTriple(), target.getChip(),
                     target.getFeatures(), target.getO()),
      target(target), toolkitPath(targetOptions.getToolkitPath()),
      fileList(targetOptions.getBitcodeFiles()) {

  // If `targetOptions` has an empty toolkitPath use `getROCMPath`
  if (toolkitPath.empty())
    toolkitPath = getROCMPath();

  // Append the files in the target attribute.
  if (ArrayAttr files = target.getLink())
    for (Attribute attr : files.getValue())
      if (auto file = dyn_cast<StringAttr>(attr))
        fileList.push_back(file.str());

  // Append standard ROCm device bitcode libraries to the files to be loaded.
  (void)appendStandardLibs();
}

void SerializeGPUModuleBase::init() {
  static llvm::once_flag initializeBackendOnce;
  llvm::call_once(initializeBackendOnce, []() {
  // If the `AMDGPU` LLVM target was built, initialize it.
#if MLIR_ROCM_CONVERSIONS_ENABLED == 1
    LLVMInitializeAMDGPUTarget();
    LLVMInitializeAMDGPUTargetInfo();
    LLVMInitializeAMDGPUTargetMC();
    LLVMInitializeAMDGPUAsmParser();
    LLVMInitializeAMDGPUAsmPrinter();
#endif
  });
}

ROCDLTargetAttr SerializeGPUModuleBase::getTarget() const { return target; }

StringRef SerializeGPUModuleBase::getToolkitPath() const { return toolkitPath; }

ArrayRef<std::string> SerializeGPUModuleBase::getFileList() const {
  return fileList;
}

LogicalResult SerializeGPUModuleBase::appendStandardLibs() {
  StringRef pathRef = getToolkitPath();
  if (pathRef.size()) {
    SmallVector<char, 256> path;
    path.insert(path.begin(), pathRef.begin(), pathRef.end());
    llvm::sys::path::append(path, "amdgcn", "bitcode");
    pathRef = StringRef(path.data(), path.size());
    if (!llvm::sys::fs::is_directory(pathRef)) {
      getOperation().emitRemark() << "ROCm amdgcn bitcode path: " << pathRef
                                  << " does not exist or is not a directory.";
      return failure();
    }
    StringRef isaVersion =
        llvm::AMDGPU::getArchNameAMDGCN(llvm::AMDGPU::parseArchAMDGCN(chip));
    isaVersion.consume_front("gfx");
    return getCommonBitcodeLibs(fileList, path, isaVersion, target.hasWave64(),
                                target.hasDaz(), target.hasFiniteOnly(),
                                target.hasUnsafeMath(), target.hasFastMath(),
                                target.hasCorrectSqrt(), target.getAbi());
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

LogicalResult
SerializeGPUModuleBase::handleBitcodeFile(llvm::Module &module,
                                          llvm::TargetMachine &targetMachine) {
  // Some ROCM builds don't strip this like they should
  if (auto *openclVersion = module.getNamedMetadata("opencl.ocl.version"))
    module.eraseNamedMetadata(openclVersion);
  // Stop spamming us with clang version numbers
  if (auto *ident = module.getNamedMetadata("llvm.ident"))
    module.eraseNamedMetadata(ident);
  return success();
}

// Get the paths of ROCm device libraries. Function adapted from:
// https://github.com/llvm/llvm-project/blob/main/clang/lib/Driver/ToolChains/AMDGPU.cpp
LogicalResult SerializeGPUModuleBase::getCommonBitcodeLibs(
    llvm::SmallVector<std::string> &libs, SmallVector<char, 256> &libPath,
    StringRef isaVersion, bool wave64, bool daz, bool finiteOnly,
    bool unsafeMath, bool fastMath, bool correctSqrt, StringRef abiVer) {
  auto addLib = [&](StringRef path) -> bool {
    if (!llvm::sys::fs::is_regular_file(path)) {
      getOperation().emitRemark() << "Bitcode library path: " << path
                                  << " does not exist or is not a file.\n";
      return true;
    }
    libs.push_back(path.str());
    return false;
  };
  auto optLib = [](StringRef name, bool on) -> Twine {
    return name + (on ? "_on" : "_off");
  };
  auto getLibPath = [&libPath](Twine lib) {
    auto baseSize = libPath.size();
    llvm::sys::path::append(libPath, lib + ".bc");
    std::string path(StringRef(libPath.data(), libPath.size()).str());
    libPath.truncate(baseSize);
    return path;
  };

  // Add ROCm device libraries. Fail if any of the libraries is not found.
  if (addLib(getLibPath("ocml")) || addLib(getLibPath("ockl")) ||
      addLib(getLibPath(optLib("oclc_daz_opt", daz))) ||
      addLib(getLibPath(optLib("oclc_unsafe_math", unsafeMath || fastMath))) ||
      addLib(getLibPath(optLib("oclc_finite_only", finiteOnly || fastMath))) ||
      addLib(getLibPath(optLib("oclc_correctly_rounded_sqrt", correctSqrt))) ||
      addLib(getLibPath(optLib("oclc_wavefrontsize64", wave64))) ||
      addLib(getLibPath("oclc_isa_version_" + isaVersion)))
    return failure();
  if (abiVer.size() && addLib(getLibPath("oclc_abi_version_" + abiVer)))
    return failure();
  return success();
}

std::optional<SmallVector<char, 0>>
SerializeGPUModuleBase::assembleIsa(StringRef isa) {
  auto loc = getOperation().getLoc();

  StringRef targetTriple = this->triple;

  SmallVector<char, 0> result;
  llvm::raw_svector_ostream os(result);

  llvm::Triple triple(llvm::Triple::normalize(targetTriple));
  std::string error;
  const llvm::Target *target =
      llvm::TargetRegistry::lookupTarget(triple.normalize(), error);
  if (!target) {
    emitError(loc, Twine("failed to lookup target: ") + error);
    return std::nullopt;
  }

  llvm::SourceMgr srcMgr;
  srcMgr.AddNewSourceBuffer(llvm::MemoryBuffer::getMemBuffer(isa), SMLoc());

  const llvm::MCTargetOptions mcOptions;
  std::unique_ptr<llvm::MCRegisterInfo> mri(
      target->createMCRegInfo(targetTriple));
  std::unique_ptr<llvm::MCAsmInfo> mai(
      target->createMCAsmInfo(*mri, targetTriple, mcOptions));
  mai->setRelaxELFRelocations(true);
  std::unique_ptr<llvm::MCSubtargetInfo> sti(
      target->createMCSubtargetInfo(targetTriple, chip, features));

  llvm::MCContext ctx(triple, mai.get(), mri.get(), sti.get(), &srcMgr,
                      &mcOptions);
  std::unique_ptr<llvm::MCObjectFileInfo> mofi(target->createMCObjectFileInfo(
      ctx, /*PIC=*/false, /*LargeCodeModel=*/false));
  ctx.setObjectFileInfo(mofi.get());

  SmallString<128> cwd;
  if (!llvm::sys::fs::current_path(cwd))
    ctx.setCompilationDir(cwd);

  std::unique_ptr<llvm::MCStreamer> mcStreamer;
  std::unique_ptr<llvm::MCInstrInfo> mcii(target->createMCInstrInfo());

  llvm::MCCodeEmitter *ce = target->createMCCodeEmitter(*mcii, ctx);
  llvm::MCAsmBackend *mab = target->createMCAsmBackend(*sti, *mri, mcOptions);
  mcStreamer.reset(target->createMCObjectStreamer(
      triple, ctx, std::unique_ptr<llvm::MCAsmBackend>(mab),
      mab->createObjectWriter(os), std::unique_ptr<llvm::MCCodeEmitter>(ce),
      *sti, mcOptions.MCRelaxAll, mcOptions.MCIncrementalLinkerCompatible,
      /*DWARFMustBeAtTheEnd*/ false));
  mcStreamer->setUseAssemblerInfoForParsing(true);

  std::unique_ptr<llvm::MCAsmParser> parser(
      createMCAsmParser(srcMgr, ctx, *mcStreamer, *mai));
  std::unique_ptr<llvm::MCTargetAsmParser> tap(
      target->createMCAsmParser(*sti, *parser, *mcii, mcOptions));

  if (!tap) {
    emitError(loc, "assembler initialization error");
    return {};
  }

  parser->setTargetParser(*tap);
  parser->Run(false);

  return result;
}

#ifdef MLIR_GPU_AMDGPU_TARGET_ENABLED
#include "llvm/Support/Program.h"

#define DEBUG_TYPE "serialize-to-object"

namespace {
class SerializeToHSA : public SerializeGPUModuleBase {
public:
  using SerializeGPUModuleBase::SerializeGPUModuleBase;

  // Create the HSACO object.
  std::optional<SmallVector<char, 0>> createHsaco(SmallVector<char, 0> &&ptx);

  std::optional<SmallVector<char, 0>>
  moduleToObject(llvm::Module &llvmModule,
                 llvm::TargetMachine &targetMachine) override;
};
} // namespace

std::optional<SmallVector<char, 0>>
SerializeToHSA::createHsaco(SmallVector<char, 0> &&ptx) {
  SmallVector<char, 0> isaBinary = std::move(ptx);
  auto loc = getOperation().getLoc();

  // Save the ISA binary to a temp file.
  int tempIsaBinaryFd = -1;
  SmallString<128> tempIsaBinaryFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "o", tempIsaBinaryFd,
                                         tempIsaBinaryFilename)) {
    emitError(loc, "temporary file for ISA binary creation error");
    return {};
  }
  llvm::FileRemover cleanupIsaBinary(tempIsaBinaryFilename);
  llvm::raw_fd_ostream tempIsaBinaryOs(tempIsaBinaryFd, true);
  tempIsaBinaryOs << StringRef(isaBinary.data(), isaBinary.size());
  tempIsaBinaryOs.close();

  // Create a temp file for HSA code object.
  int tempHsacoFD = -1;
  SmallString<128> tempHsacoFilename;
  if (llvm::sys::fs::createTemporaryFile("kernel", "hsaco", tempHsacoFD,
                                         tempHsacoFilename)) {
    emitError(loc, "temporary file for HSA code object creation error");
    return {};
  }
  llvm::FileRemover cleanupHsaco(tempHsacoFilename);

  llvm::SmallString<32> lldPath(toolkitPath);
  llvm::sys::path::append(lldPath, "llvm", "bin", "ld.lld");
  int lldResult = llvm::sys::ExecuteAndWait(
      lldPath,
      {"ld.lld", "-shared", tempIsaBinaryFilename, "-o", tempHsacoFilename});
  if (lldResult != 0) {
    emitError(loc, "lld invocation error");
    return {};
  }

  // Load the HSA code object.
  auto hsacoFile = openInputFile(tempHsacoFilename);
  if (!hsacoFile) {
    emitError(loc, "read HSA code object from temp file error");
    return {};
  }

  StringRef buffer = hsacoFile->getBuffer();

  return SmallVector<char, 0>(buffer.begin(), buffer.end());
}

std::optional<SmallVector<char, 0>>
SerializeToHSA::moduleToObject(llvm::Module &llvmModule,
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

  std::optional<SmallVector<char, 0>> assembledIsa =
      assembleIsa(serializedISA.value());

  if (!assembledIsa) {
    getOperation().emitError() << "Failed during ISA assembling.";
    return std::nullopt;
  }

  return createHsaco(std::move(assembledIsa.value()));
}
#endif // MLIR_GPU_AMDGPU_TARGET_ENABLED

std::optional<SmallVector<char, 0>> ROCDLTargetAttrImpl::serializeToObject(
    Attribute attribute, Operation *module,
    const gpu::TargetOptions &options) const {
  assert(module && "The module must be non null.");
  if (!module)
    return std::nullopt;
  if (!mlir::isa<gpu::GPUModuleOp>(module)) {
    module->emitError("Module must be a GPU module.");
    return std::nullopt;
  }
#ifdef MLIR_GPU_AMDGPU_TARGET_ENABLED
  SerializeToHSA serializer(*module, cast<ROCDLTargetAttr>(attribute), options);
  serializer.init();
  return serializer.run();
#else
  return SmallVector<char, 0>{};
#endif
}
