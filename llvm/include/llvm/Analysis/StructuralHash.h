//=- StructuralHash.h - Structural Hash Printing --*- C++ -*---------------0-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the StructuralHashPrinterPass
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_ANALYSIS_STRUCTURALHASH_H
#define LLVM_ANALYSIS_STRUCTURALHASH_H

#include "llvm/IR/PassManager.h"

namespace llvm {

/// Printer pass for  StructuralHashes
class StructuralHashPrinterPass
    : public PassInfoMixin<StructuralHashPrinterPass> {
  raw_ostream &OS;

public:
  explicit StructuralHashPrinterPass(raw_ostream &OS) : OS(OS) {}

  PreservedAnalyses run(Module &M, ModuleAnalysisManager &MAM);
};

} // namespace llvm

#endif // LLVM_ANALYSIS_STRUCTURALHASH_H
