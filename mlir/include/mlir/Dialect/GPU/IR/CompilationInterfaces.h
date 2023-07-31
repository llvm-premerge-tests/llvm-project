//===-- CompilationInterfaces.h - GPU compilation interfaces ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for GPU target attributes.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_GPU_IR_COMPILATIONINTERFACES_H
#define MLIR_DIALECT_GPU_IR_COMPILATIONINTERFACES_H

#include "mlir/IR/Attributes.h"
#include "mlir/IR/Location.h"

namespace mlir {
namespace gpu {
/// This class serves as an opaque interface for passing options to the
/// `TargetAttrInterface` methods. Users of this class must implement the
/// `classof` method as well as using the macros `MLIR_*_EXPLICIT_TYPE_ID` to
/// ensure type safeness.
class TargetOptions {
public:
  /// Constructor initializing the toolkit path and the list of bitcode files.
  TargetOptions(StringRef toolkitPath = {},
                ArrayRef<std::string> bitcodeFiles = {});

  /// Returns the typeID.
  TypeID getTypeID() const;

  /// Returns the toolkit path.
  StringRef getToolkitPath() const;

  /// Returns the bitcode files to link to.
  ArrayRef<std::string> getBitcodeFiles() const;

protected:
  /// Derived classes must use this constructor to initialize `typeID` to the
  /// appropiate value: ie. `TargetOptions(TypeID::get<DerivedClass>())`.
  TargetOptions(TypeID typeID, StringRef toolkitPath = {},
                ArrayRef<std::string> bitcodeFiles = {});

  /// Path to the target toolkit.
  std::string toolkitPath;

  /// List of files to link with the LLVM module.
  SmallVector<std::string> bitcodeFiles;

private:
  TypeID typeID;
};
} // namespace gpu
} // namespace mlir

MLIR_DECLARE_EXPLICIT_TYPE_ID(::mlir::gpu::TargetOptions)

#include "mlir/Dialect/GPU/IR/CompilationAttrInterfaces.h.inc"

#endif
