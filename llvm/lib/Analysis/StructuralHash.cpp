//===- StructuralHash.cpp - Function Hash Printing ------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the StructuralHashPrinterPass which is used to show
// the structural hash of all functions in a module and the module itself.
//
//===----------------------------------------------------------------------===//

#include "llvm/Analysis/StructuralHash.h"
#include "llvm/IR/StructuralHash.h"
#include "llvm/Support/CommandLine.h"

using namespace llvm;

cl::opt<bool> EnableDetailedStructuralHashing(
    "enable-detailed-structural-hash", cl::Hidden, cl::init(false),
    cl::desc("Whether or not to enable detailed structural hashing."));

PreservedAnalyses StructuralHashPrinterPass::run(Module &M,
                                                 ModuleAnalysisManager &MAM) {
  OS << "Module Hash: "
     << Twine::utohexstr(StructuralHash(M, EnableDetailedStructuralHashing))
     << "\n";
  for (Function &F : M) {
    OS << "Function " << F.getName() << " Hash: "
       << Twine::utohexstr(StructuralHash(F, EnableDetailedStructuralHashing))
       << "\n";
  }
  return PreservedAnalyses::all();
}
