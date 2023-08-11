//===- LegalizeForExport.cpp - Prepare for translation to LLVM IR ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/LegalizeForExport.h"

#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {
namespace LLVM {
#define GEN_PASS_DEF_LLVMLEGALIZEFOREXPORT
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace LLVM
} // namespace mlir

using namespace mlir;

namespace {
struct LegalizeForExportPass
    : public LLVM::impl::LLVMLegalizeForExportBase<LegalizeForExportPass> {
  void runOnOperation() override {
    LLVM::ensureDistinctSuccessors(getOperation());
  }
};
} // namespace

std::unique_ptr<Pass> LLVM::createLegalizeForExportPass() {
  return std::make_unique<LegalizeForExportPass>();
}
