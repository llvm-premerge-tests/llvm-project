//===- AIXLinkerImportFile.cpp -AIX linker Import file implement-*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the AIX Linker Import File interface.
//
//===----------------------------------------------------------------------===//

#include "llvm/Object/AIXLinkerImportFile.h"
#include "llvm/BinaryFormat/Magic.h"

namespace llvm {
namespace object {

AIXLinkerImportFile::~AIXLinkerImportFile() = default;
Expected<std::unique_ptr<AIXLinkerImportFile>>
AIXLinkerImportFile::create(MemoryBufferRef Source) {
  return std::make_unique<AIXLinkerImportFile>(
      file_magic::aix_linker_import_file, Source);
}

} // namespace object
} // namespace llvm
