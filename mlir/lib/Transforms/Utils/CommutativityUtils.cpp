//===- CommutativityUtils.cpp - Commutativity utilities ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a commutativity utility pattern and a function to
// populate this pattern. The function is intended to be used inside passes to
// simplify the matching of commutative operations by fixing the order of their
// operands.
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/CommutativityUtils.h"

using namespace mlir;

AncestorKey::AncestorKey(Operation *op) {
  if (!op) {
    type = BLOCK_ARGUMENT;
  } else {
    type =
        op->hasTrait<OpTrait::ConstantLike>() ? CONSTANT_OP : NON_CONSTANT_OP;
    opName = op->getName().getStringRef();
  }
}

bool AncestorKey::operator<(const AncestorKey &key) const {
  return std::tie(type, opName) < std::tie(key.type, key.opName);
}

void CommutativeOperand::pushAncestor(Operation *op) {
  ancestorQueue.push(op);
  if (op)
    visitedAncestors.insert(op);
}

void CommutativeOperand::refreshKey() {
  if (ancestorQueue.empty())
    return;

  Operation *frontAncestor = ancestorQueue.front();
  AncestorKey frontAncestorKey(frontAncestor);
  key.push_back(frontAncestorKey);
}

void CommutativeOperand::popFrontAndPushAdjacentUnvisitedAncestors() {
  if (ancestorQueue.empty())
    return;
  Operation *frontAncestor = ancestorQueue.front();
  ancestorQueue.pop();
  if (!frontAncestor)
    return;
  for (Value operand : frontAncestor->getOperands()) {
    Operation *operandDefOp = operand.getDefiningOp();
    if (!operandDefOp || !visitedAncestors.contains(operandDefOp))
      pushAncestor(operandDefOp);
  }
}

bool CommutativeOperand::commutativeOperandComparator(
    const std::unique_ptr<CommutativeOperand> &constCommOperandA,
    const std::unique_ptr<CommutativeOperand> &constCommOperandB) {
  if (constCommOperandA->operand == constCommOperandB->operand)
    return false;

  auto &commOperandA =
      const_cast<std::unique_ptr<CommutativeOperand> &>(constCommOperandA);
  auto &commOperandB =
      const_cast<std::unique_ptr<CommutativeOperand> &>(constCommOperandB);

  // Iteratively perform the BFS's of both operands until an order among
  // them can be determined.
  unsigned keyIndex = 0;
  while (true) {
    if (commOperandA->key.size() <= keyIndex) {
      if (commOperandA->ancestorQueue.empty())
        return true;
      commOperandA->popFrontAndPushAdjacentUnvisitedAncestors();
      commOperandA->refreshKey();
    }
    if (commOperandB->key.size() <= keyIndex) {
      if (commOperandB->ancestorQueue.empty())
        return false;
      commOperandB->popFrontAndPushAdjacentUnvisitedAncestors();
      commOperandB->refreshKey();
    }
    // Try comparing the keys at the current keyIndex
    if (keyIndex < commOperandA->key.size() &&
        keyIndex < commOperandB->key.size()) {
      if (commOperandA->key[keyIndex] < commOperandB->key[keyIndex])
        return true;
      if (commOperandB->key[keyIndex] < commOperandA->key[keyIndex])
        return false;
    } else { // keyIndex exceeds one or both key sizes
      // Compare key sizes if the values at every possible keyIndex were
      // equal Both operands must have fully generated key and cannot
      // have anything in the ancestorQueue
      if (commOperandA->ancestorQueue.empty() &&
          commOperandB->ancestorQueue.empty())
        return commOperandA->key.size() < commOperandB->key.size();
    }
    keyIndex++;
  }
}

/// Sorts the operands of `op` in ascending order of the "key" associated with
/// each operand iff `op` is commutative. This is a stable sort.
///
/// After the application of this pattern, since the commutative operands now
/// have a deterministic order in which they occur in an op, the matching of
/// large DAGs becomes much simpler, i.e., requires much less number of checks
/// to be written by a user in her/his pattern matching function.
///
/// Some examples of such a sorting:
///
/// Assume that the sorting is being applied to `foo.commutative`, which is a
/// commutative op.
///
/// Example 1:
///
/// %1 = foo.const 0
/// %2 = foo.mul <block argument>, <block argument>
/// %3 = foo.commutative %1, %2
///
/// Here,
/// 1. The key associated with %1 is:
///     `{
///       {CONSTANT_OP, "foo.const"}
///      }`
/// 2. The key associated with %2 is:
///     `{
///       {NON_CONSTANT_OP, "foo.mul"},
///       {BLOCK_ARGUMENT, ""},
///       {BLOCK_ARGUMENT, ""}
///      }`
///
/// The key of %2 < the key of %1
/// Thus, the sorted `foo.commutative` is:
/// %3 = foo.commutative %2, %1
///
/// Example 2:
///
/// %1 = foo.const 0
/// %2 = foo.mul <block argument>, <block argument>
/// %3 = foo.mul %2, %1
/// %4 = foo.add %2, %1
/// %5 = foo.commutative %1, %2, %3, %4
///
/// Here,
/// 1. The key associated with %1 is:
///     `{
///       {CONSTANT_OP, "foo.const"}
///      }`
/// 2. The key associated with %2 is:
///     `{
///       {NON_CONSTANT_OP, "foo.mul"},
///       {BLOCK_ARGUMENT, ""},
///       {BLOCK_ARGUMENT, ""}
///      }`
/// 3. The key associated with %3 is:
///     `{
///       {NON_CONSTANT_OP, "foo.mul"},
///       {NON_CONSTANT_OP, "foo.mul"},
///       {CONSTANT_OP, "foo.const"},
///       {BLOCK_ARGUMENT, ""},
///       {BLOCK_ARGUMENT, ""}
///      }`
/// 4. The key associated with %4 is:
///     `{
///       {NON_CONSTANT_OP, "foo.add"},
///       {NON_CONSTANT_OP, "foo.mul"},
///       {CONSTANT_OP, "foo.const"},
///       {BLOCK_ARGUMENT, ""},
///       {BLOCK_ARGUMENT, ""}
///      }`
///
/// Thus, the sorted `foo.commutative` is:
/// %5 = foo.commutative %4, %2, %3, %1
struct SortCommutativeOperands final
    : public OpTraitRewritePattern<OpTrait::IsCommutative> {
  SortCommutativeOperands(MLIRContext *context)
      : OpTraitRewritePattern<OpTrait::IsCommutative>(context, /*benefit=*/5) {}
  LogicalResult matchAndRewrite(Operation *op,
                                PatternRewriter &rewriter) const override {
    // Populate the list of commutative operands.
    SmallVector<Value, 2> operands = op->getOperands();
    SmallVector<std::unique_ptr<CommutativeOperand>, 2> commOperands;
    for (Value operand : operands) {
      std::unique_ptr<CommutativeOperand> commOperand =
          std::make_unique<CommutativeOperand>();
      commOperand->operand = operand;
      commOperand->pushAncestor(operand.getDefiningOp());
      commOperand->refreshKey();
      commOperands.push_back(std::move(commOperand));
    }

    // Sort the operands.
    std::stable_sort(commOperands.begin(), commOperands.end(),
                     CommutativeOperand::commutativeOperandComparator);
    SmallVector<Value, 2> sortedOperands;
    for (const std::unique_ptr<CommutativeOperand> &commOperand : commOperands)
      sortedOperands.push_back(commOperand->operand);
    if (sortedOperands == operands)
      return failure();
    rewriter.updateRootInPlace(op, [&] { op->setOperands(sortedOperands); });
    return success();
  }
};

void mlir::populateCommutativityUtilsPatterns(RewritePatternSet &patterns) {
  patterns.add<SortCommutativeOperands>(patterns.getContext());
}
