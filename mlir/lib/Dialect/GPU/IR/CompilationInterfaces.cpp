//===-- CompilationInterfaces.cpp - GPU compilation interfaces --*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for GPU compilation attributes.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/CompilationInterfaces.h"

#include "llvm/Support/CommandLine.h"
#include "llvm/Support/StringSaver.h"

using namespace mlir;
using namespace mlir::gpu;

//===----------------------------------------------------------------------===//
// GPU target options
//===----------------------------------------------------------------------===//

TargetOptions::TargetOptions(StringRef toolkitPath,
                             ArrayRef<std::string> linkFiles,
                             StringRef cmdOptions,
                             CompilationTarget compilationTarget)
    : TargetOptions(TypeID::get<TargetOptions>(), toolkitPath, linkFiles,
                    cmdOptions, compilationTarget) {}

TargetOptions::TargetOptions(TypeID typeID, StringRef toolkitPath,
                             ArrayRef<std::string> linkFiles,
                             StringRef cmdOptions,
                             CompilationTarget compilationTarget)
    : toolkitPath(toolkitPath.str()), linkFiles(linkFiles),
      cmdOptions(cmdOptions.str()), compilationTarget(compilationTarget),
      typeID(typeID) {}

TypeID TargetOptions::getTypeID() const { return typeID; }

StringRef TargetOptions::getToolkitPath() const { return toolkitPath; }

ArrayRef<std::string> TargetOptions::getLinkFiles() const { return linkFiles; }

StringRef TargetOptions::getCmdOptions() const { return cmdOptions; }

std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>>
TargetOptions::tokenizeCmdOptions() const {
  std::pair<llvm::BumpPtrAllocator, SmallVector<const char *>> options;
  llvm::StringSaver stringSaver(options.first);
#ifdef _WIN32
  llvm::cl::TokenizeWindowsCommandLine(cmdOptions, stringSaver, options.second,
                                       /*MarkEOLs=*/false);
#else
  llvm::cl::TokenizeGNUCommandLine(cmdOptions, stringSaver, options.second,
                                   /*MarkEOLs=*/false);
#endif // _WIN32
  return options;
}

TargetOptions::CompilationTarget TargetOptions::getCompilationTarget() const {
  return compilationTarget;
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::TargetOptions)

#include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.cpp.inc"
