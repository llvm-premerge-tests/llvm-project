//===- CommutativityUtils.cpp - Commutative Operand Comparison unit tests -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/CommutativityUtils.h"
#include "mlir/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "gtest/gtest.h"
#include <cstddef>

using namespace mlir;

static Operation *createOp(MLIRContext *context, StringRef name,
                           TypeRange types = std::nullopt) {
  context->allowUnregisteredDialects();
  return Operation::create(
      UnknownLoc::get(context), OperationName(name, context), types,
      std::nullopt, std::nullopt, /*properties=*/nullptr, std::nullopt, 0);
}

// The following test case is based on this example
// %0 = dummy1 <block-argument> <block-argument>
// %1 = dummy2 <block-argument> <block-argument>
// %commutative = %0, %1
//
// The test case compares the commutative operands dummy1 and dummy2.
// dummy1's key is complete and its ancestorQueue is empty while dummy2's
// key only has one element and its ancestorQueue has the remaining 2
// elements.
//
// dummy1 is the smaller operand because its name is lexicographically
// smaller than dummy2
TEST(CommutativityUtilsTest, TestOneEmptyAncestorQueue) {
  MLIRContext context;
  Builder builder(&context);
  Operation *dummy1 = createOp(&context, "dummy1", builder.getIntegerType(32));
  Operation *dummy2 = createOp(&context, "dummy2", builder.getIntegerType(32));

  std::unique_ptr<CommutativeOperand> commOperandA =
      std::make_unique<CommutativeOperand>();
  commOperandA->operand = dummy1->getOpResult(0);
  AncestorKey commOperandAKey0(dummy1);
  AncestorKey commOperandAKey1(nullptr);
  AncestorKey commOperandAKey2(nullptr);
  commOperandA->key.push_back(commOperandAKey0);
  commOperandA->key.push_back(commOperandAKey1);
  commOperandA->key.push_back(commOperandAKey2);

  std::unique_ptr<CommutativeOperand> commOperandB =
      std::make_unique<CommutativeOperand>();
  commOperandB->operand = dummy2->getOpResult(0);
  AncestorKey commOperandBKey0(dummy2);
  commOperandB->key.push_back(commOperandBKey0);
  commOperandB->ancestorQueue.push(nullptr);
  commOperandB->ancestorQueue.push(nullptr);

  EXPECT_TRUE(CommutativeOperand::commutativeOperandComparator(commOperandA,
                                                               commOperandB));
}
