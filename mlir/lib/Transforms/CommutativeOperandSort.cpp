//===- CommutativeOperandSort.cpp - Commutative Operand Sorting Pass ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This transformation pass sorts operands of commutative operations in a
// deterministic manner so they can be compared and removed in CSE.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/Passes.h"

#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/CommutativityUtils.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"

namespace mlir {
#define GEN_PASS_DEF_COMMUTATIVEOPERANDSORT
#include "mlir/Transforms/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct CommutativeOperandSortPass
    : public impl::CommutativeOperandSortBase<CommutativeOperandSortPass> {
  CommutativeOperandSortPass() = default;

  void runOnOperation() override {
    // Populate pattern
    RewritePatternSet patterns(&getContext());
    populateCommutativityUtilsPatterns(patterns);

    if (failed(
            applyPatternsAndFoldGreedily(getOperation(), std::move(patterns))))
      signalPassFailure();
  }
};
} // end anonymous namespace

// Create a Commutative operand sorting pass
std::unique_ptr<Pass> mlir::createCommutativeOperandSortPass() {
  return std::make_unique<CommutativeOperandSortPass>();
}
