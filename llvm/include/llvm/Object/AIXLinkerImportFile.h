//===- AIXLinkerImportFile.h - AIX linker Import file implement-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the AIX Linker Import File interface.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_AIX_LINKER_IMPORT_FILE_H
#define LLVM_AIX_LINKER_IMPORT_FILE_H

#include "llvm/Object/Binary.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/MemoryBufferRef.h"

namespace llvm {
namespace object {

// The class AIXLinkerImportFile, which is not a symbolic file at the moment.
// The command 'llvm-nm -export-symbols' exports an empty symbol for AIX linker
// import file that depends on the AIXLinkerImportFile. Furthermore, if we
// decide to get symbols from AIX linker import file and derive
// AIXLinkerImportFile from the SymbolicFile class, the llvm-nm tool will also
// need to be updated accordingly.

class AIXLinkerImportFile : public Binary {
public:
  AIXLinkerImportFile(unsigned int Type, MemoryBufferRef Source)
      : Binary(Type, Source) {}
  ~AIXLinkerImportFile();
  static Expected<std::unique_ptr<AIXLinkerImportFile>>
  create(MemoryBufferRef Source);
};
} // namespace object
} // namespace llvm

#endif
