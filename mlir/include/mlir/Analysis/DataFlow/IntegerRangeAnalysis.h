//===-IntegerRangeAnalysis.h - Integer range analysis -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file declares the dataflow analysis class for integer range inference
// so that it can be used in transformations over the `arith` dialect such as
// branch elimination or signed->unsigned rewriting
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"
#include "mlir/Interfaces/InferIntRangeInterface.h"

namespace mlir {
namespace dataflow {

/// This lattice value represents the integer range of an SSA value.
class IntegerValueRange {
public:
  /// Create a maximal range ([0, uint_max(t)] / [int_min(t), int_max(t)])
  /// range that is used to mark the value as unable to be analyzed further,
  /// where `t` is the type of `value`.
  static IntegerValueRange getPessimisticValue(Value value);

  /// Create an integer value range lattice value.
  IntegerValueRange(ConstantIntRanges value) : value(std::move(value)) {}

  /// Get the known integer value range.
  const ConstantIntRanges &getValue() const { return value; }

  /// Compare two ranges.
  bool operator==(const IntegerValueRange &rhs) const {
    return value == rhs.value;
  }

  /// Take the union of two ranges.
  static IntegerValueRange join(const IntegerValueRange &lhs,
                                const IntegerValueRange &rhs) {
    return lhs.value.rangeUnion(rhs.value);
  }
  static IntegerValueRange meet(const IntegerValueRange &lhs,
                                const IntegerValueRange &rhs) {
    return lhs.value.intersection(rhs.value);
  }

  /// Print the integer value range.
  void print(raw_ostream &os) const { os << value; }

private:
  /// The known integer value range.
  ConstantIntRanges value;
};

/// This lattice element represents the integer value range of an SSA value.
/// When this lattice is updated, it automatically updates the constant value
/// of the SSA value (if the range can be narrowed to one).
class IntegerValueRangeState : public OptimisticSparseState<IntegerValueRange> {
public:
  using OptimisticSparseState::OptimisticSparseState;
  using ElementT =
      SparseElement<IntegerValueRangeState, SingleStateElement>;
};

/// Integer range analysis determines the integer value range of SSA values
/// using operations that define `InferIntRangeInterface` and also sets the
/// range of iteration indices of loops with known bounds.
class IntegerRangeAnalysis
    : public SparseDataFlowAnalysis<IntegerValueRangeState> {
public:
  using SparseDataFlowAnalysis::SparseDataFlowAnalysis;

  /// Visit an operation. Invoke the transfer function on each operation that
  /// implements `InferIntRangeInterface`.
  void
  visitOperation(Operation *op,
                 ArrayRef<const IntegerValueRangeState *> operands,
                 ArrayRef<IntegerValueRangeState::ElementT *> results) override;

  /// Visit block arguments or operation results of an operation with region
  /// control-flow for which values are not defined by region control-flow. This
  /// function calls `InferIntRangeInterface` to provide values for block
  /// arguments or tries to reduce the range on loop induction variables with
  /// known bounds.
  void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<IntegerValueRangeState::ElementT *> argLattices,
      unsigned firstIndex) override;
};

class IntegerRangeToConstant : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  LogicalResult initialize(Operation *top) override;
  LogicalResult visit(ProgramPoint point) override;

  bool staticallyProvides(TypeID stateID, ProgramPoint point) const override {
    return stateID == TypeID::get<ConstantValueState>() && point.is<Value>();
  }
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_INTEGERANGEANALYSIS_H
