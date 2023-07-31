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

using namespace mlir;
using namespace mlir::gpu;

//===----------------------------------------------------------------------===//
// GPU Compilation options
//===----------------------------------------------------------------------===//

TargetOptions::TargetOptions(StringRef toolkitPath,
                             ArrayRef<std::string> bitcodeFiles)
    : TargetOptions(TypeID::get<TargetOptions>(), toolkitPath, bitcodeFiles) {}

TargetOptions::TargetOptions(TypeID typeID, StringRef toolkitPath,
                             ArrayRef<std::string> bitcodeFiles)
    : toolkitPath(toolkitPath.str()), bitcodeFiles(bitcodeFiles),
      typeID(typeID) {}

TypeID TargetOptions::getTypeID() const { return typeID; }

StringRef TargetOptions::getToolkitPath() const { return toolkitPath; }

ArrayRef<std::string> TargetOptions::getBitcodeFiles() const {
  return bitcodeFiles;
}

MLIR_DEFINE_EXPLICIT_TYPE_ID(::mlir::gpu::TargetOptions)

#include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.cpp.inc"
