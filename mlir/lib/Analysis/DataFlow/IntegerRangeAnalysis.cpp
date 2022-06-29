//===- IntegerRangeAnalysis.cpp - Integer range analysis --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the dataflow analysis class for integer range inference
// which is used in transformations over the `arith` dialect such as
// branch elimination or signed->unsigned rewriting
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "int-range-analysis"

using namespace mlir;
using namespace mlir::dataflow;

IntegerValueRange IntegerValueRange::getPessimisticValue(Value value) {
  unsigned width = ConstantIntRanges::getStorageBitwidth(value.getType());
  APInt umin = APInt::getMinValue(width);
  APInt umax = APInt::getMaxValue(width);
  APInt smin = width != 0 ? APInt::getSignedMinValue(width) : umin;
  APInt smax = width != 0 ? APInt::getSignedMaxValue(width) : umax;
  return {{umin, umax, smin, smax}};
}

void IntegerRangeAnalysis::visitOperation(
    Operation *op, ArrayRef<const IntegerValueRangeState *> operands,
    ArrayRef<IntegerValueRangeState::ElementT *> results) {
  // Ignore non-integer outputs - return early if the op has no scalar
  // integer results
  bool hasIntegerResult = false;
  for (auto it : llvm::zip(results, op->getResults())) {
    if (std::get<1>(it).getType().isIntOrIndex()) {
      hasIntegerResult = true;
    } else {
      std::get<0>(it)->update(this, [](IntegerValueRangeState *state) {
        return state->markPessimisticFixpoint();
      });
    }
  }
  if (!hasIntegerResult)
    return;

  auto inferrable = dyn_cast<InferIntRangeInterface>(op);
  if (!inferrable)
    return markAllPessimisticFixpoint(results);

  LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
  SmallVector<ConstantIntRanges> argRanges(
      llvm::map_range(operands, [](const IntegerValueRangeState *val) {
        return val->getValue().getValue();
      }));

  auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
    auto result = v.dyn_cast<OpResult>();
    if (!result)
      return;
    assert(llvm::find(op->getResults(), result) != op->result_end());

    LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
    results[result.getResultNumber()]->update(
        this, [&](IntegerValueRangeState *state) {
          Optional<IntegerValueRange> oldRange;
          if (!state->isUninitialized())
            oldRange = state->getValue();

          ChangeResult changed = state->join(attrs);

          // Catch loop results with loop variant bounds and conservatively make
          // them [-inf, inf] so we don't circle around infinitely often
          // (because the dataflow analysis in MLIR doesn't attempt to work out
          // trip counts and often can't).
          bool isYieldedResult = llvm::any_of(v.getUsers(), [](Operation *op) {
            return op->hasTrait<OpTrait::IsTerminator>();
          });
          if (isYieldedResult && oldRange.hasValue() &&
              !(state->getValue() == *oldRange)) {
            LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
            changed |= state->markPessimisticFixpoint();
          }
          return changed;
        });
  };

  inferrable.inferResultRanges(argRanges, joinCallback);
}

void IntegerRangeAnalysis::visitNonControlFlowArguments(
    Operation *op, const RegionSuccessor &successor,
    ArrayRef<IntegerValueRangeState::ElementT *> argLattices,
    unsigned firstIndex) {
  if (auto inferrable = dyn_cast<InferIntRangeInterface>(op)) {
    LLVM_DEBUG(llvm::dbgs() << "Inferring ranges for " << *op << "\n");
    SmallVector<ConstantIntRanges> argRanges(
        llvm::map_range(op->getOperands(), [&](Value value) {
          return getLatticeElementFor(op, value)->getValue().getValue();
        }));

    auto joinCallback = [&](Value v, const ConstantIntRanges &attrs) {
      auto arg = v.dyn_cast<BlockArgument>();
      if (!arg)
        return;
      if (llvm::find(successor.getSuccessor()->getArguments(), arg) ==
          successor.getSuccessor()->args_end())
        return;

      LLVM_DEBUG(llvm::dbgs() << "Inferred range " << attrs << "\n");
      argLattices[arg.getArgNumber()]->update(
          this, [&](IntegerValueRangeState *state) {
            Optional<IntegerValueRange> oldRange;
            if (!state->isUninitialized())
              oldRange = state->getValue();

            ChangeResult changed = state->join(attrs);

            // Catch loop results with loop variant bounds and conservatively
            // make them [-inf, inf] so we don't circle around infinitely often
            // (because the dataflow analysis in MLIR doesn't attempt to work
            // out trip counts and often can't).
            bool isYieldedValue = llvm::any_of(v.getUsers(), [](Operation *op) {
              return op->hasTrait<OpTrait::IsTerminator>();
            });
            if (isYieldedValue && oldRange &&
                !(state->getValue() == *oldRange)) {
              LLVM_DEBUG(llvm::dbgs() << "Loop variant loop result detected\n");
              changed |= state->markPessimisticFixpoint();
            }
            return changed;
          });
    };

    inferrable.inferResultRanges(argRanges, joinCallback);
    return;
  }

  /// Given the results of getConstant{Lower,Upper}Bound() or getConstantStep()
  /// on a LoopLikeInterface return the lower/upper bound for that result if
  /// possible.
  auto getLoopBoundFromFold = [&](Optional<OpFoldResult> loopBound,
                                  Type boundType, bool getUpper) {
    unsigned int width = ConstantIntRanges::getStorageBitwidth(boundType);
    if (loopBound.hasValue()) {
      if (loopBound->is<Attribute>()) {
        if (auto bound =
                loopBound->get<Attribute>().dyn_cast_or_null<IntegerAttr>())
          return bound.getValue();
      } else if (auto value = loopBound->dyn_cast<Value>()) {
        const IntegerValueRangeState *state = getLatticeElementFor(op, value);
        return getUpper ? state->getValue().getValue().smax()
                        : state->getValue().getValue().smin();
      }
    }
    // Given the results of getConstant{Lower,Upper}Bound()
    // or getConstantStep() on a LoopLikeInterface return the lower/upper
    // bound
    return getUpper ? APInt::getSignedMaxValue(width)
                    : APInt::getSignedMinValue(width);
  };

  // Infer bounds for loop arguments that have static bounds
  if (auto loop = dyn_cast<LoopLikeOpInterface>(op)) {
    Optional<Value> iv = loop.getSingleInductionVar();
    if (!iv) {
      return SparseDataFlowAnalysis ::visitNonControlFlowArguments(
          op, successor, argLattices, firstIndex);
    }
    Optional<OpFoldResult> lowerBound = loop.getSingleLowerBound();
    Optional<OpFoldResult> upperBound = loop.getSingleUpperBound();
    Optional<OpFoldResult> step = loop.getSingleStep();
    APInt min = getLoopBoundFromFold(lowerBound, iv->getType(), false);
    APInt max = getLoopBoundFromFold(upperBound, iv->getType(), true);
    // Assume positivity for uniscoverable steps by way of getUpper = true.
    APInt stepVal = getLoopBoundFromFold(step, iv->getType(), true);

    if (stepVal.isNegative()) {
      std::swap(min, max);
    } else {
      // Correct the upper bound by subtracting 1 so that it becomes a <=
      // bound, because loops do not generally include their upper bound.
      max -= 1;
    }

    auto *ivEntry = getLatticeElement(*iv);
    auto ivRange = ConstantIntRanges::fromSigned(min, max);
    return ivEntry->update(this, [&ivRange](IntegerValueRangeState *state) {
      return state->join(ivRange);
    });
  }

  return SparseDataFlowAnalysis::visitNonControlFlowArguments(
      op, successor, argLattices, firstIndex);
}

LogicalResult IntegerRangeToConstant::initialize(Operation *top) {
  auto visitValues = [this](ValueRange values) {
    for (Value value : values)
      (void)visit(value);
  };
  top->walk([&](Operation *op) {
    visitValues(op->getResults());
    for (Region &region : op->getRegions())
      for (Block &block : region)
        visitValues(block.getArguments());
  });
  return success();
}

LogicalResult IntegerRangeToConstant::visit(ProgramPoint point) {
  auto value = point.get<Value>();
  auto *rangeState = getOrCreateFor<IntegerValueRangeState>(value, value);
  if (rangeState->isUninitialized())
    return success();

  update<ConstantValueState>(value, [&](ConstantValueState *state) {
    const ConstantIntRanges &range = rangeState->getValue().getValue();
    // Try to narrow to a constant.
    Optional<APInt> constant = range.getConstantValue();
    if (!constant)
      return state->markPessimisticFixpoint();

    // Find a dialect to materialize the constant.
    Dialect *dialect;
    if (Operation *op = value.getDefiningOp())
      dialect = op->getDialect();
    else
      dialect = value.getParentRegion()->getParentOp()->getDialect();

    Attribute attr = IntegerAttr::get(value.getType(), *constant);
    return state->join(ConstantValue(attr, dialect));
  });
  return success();
}
