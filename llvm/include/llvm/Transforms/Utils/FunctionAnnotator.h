//===-- FunctionAnnotator.h - FunctionAnnotator ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// This file defines the FunctionAnnotator class which is used in phase
// ordering.
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_FUNCTIONANNOTATOR_H
#define LLVM_TRANSFORMS_FUNCTIONANNOTATOR_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
class FunctionAnnotator : public PassInfoMixin<FunctionAnnotator> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_FUNCTIONANNOTATOR_H
