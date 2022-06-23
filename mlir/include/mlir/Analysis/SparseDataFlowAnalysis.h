//===- SparseDataFlowAnalysis.h - Sparse data-flow analysis ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements sparse data-flow analysis using the data-flow analysis
// framework. The analysis is forward and conditional and uses the results of
// dead code analysis to prune dead code during the analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_SPARSEDATAFLOWANALYSIS_H
#define MLIR_ANALYSIS_SPARSEDATAFLOWANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Interfaces/CallInterfaces.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {

//===----------------------------------------------------------------------===//
// AbstractSparseLattice
//===----------------------------------------------------------------------===//

/// This class represents an abstract lattice. A lattice contains information
/// about an SSA value and is what's propagated across the IR by sparse
/// data-flow analysis.
class AbstractSparseLattice : public AnalysisState {
public:
  /// Lattices can only be created for values.
  AbstractSparseLattice(Value value) : AnalysisState(value) {}

  /// Join the information contained in 'rhs' into this lattice. Returns
  /// if the value of the lattice changed.
  virtual ChangeResult join(const AbstractSparseLattice &rhs) = 0;

  /// Returns true if the lattice element is at fixpoint and further calls to
  /// `join` will not update the value of the element.
  virtual bool isAtFixpoint() const = 0;

  /// Mark the lattice element as having reached a pessimistic fixpoint. This
  /// means that the lattice may potentially have conflicting value states, and
  /// only the most conservative value should be relied on.
  virtual ChangeResult markPessimisticFixpoint() = 0;

  /// When the lattice gets updated, propagate an update to users of the value
  /// using its use-def chain to subscribed analyses.
  void onUpdate(DataFlowSolver *solver) const override;

  /// Subscribe an analysis to updates of the lattice. When the lattice changes,
  /// subscribed analyses are re-invoked on all users of the value. This is
  /// more efficient than relying on the dependency map.
  void useDefSubscribe(DataFlowAnalysis *analysis) {
    useDefSubscribers.insert(analysis);
  }

private:
  /// A set of analyses that should be updated when this lattice changes.
  SetVector<DataFlowAnalysis *, SmallVector<DataFlowAnalysis *, 4>,
            SmallPtrSet<DataFlowAnalysis *, 4>>
      useDefSubscribers;
};

//===----------------------------------------------------------------------===//
// Lattice
//===----------------------------------------------------------------------===//

/// This class represents a lattice holding a specific value of type `ValueT`.
/// Lattice values (`ValueT`) are required to adhere to the following:
///
///   * static ValueT join(const ValueT &lhs, const ValueT &rhs);
///     - This method conservatively joins the information held by `lhs`
///       and `rhs` into a new value. This method is required to be monotonic.
///   * bool operator==(const ValueT &rhs) const;
///
template <typename ValueT>
class Lattice : public AbstractSparseLattice {
public:
  using AbstractSparseLattice::AbstractSparseLattice;

  /// Get a lattice element with a known value.
  Lattice(const ValueT &knownValue = ValueT())
      : AbstractSparseLattice(Value()), knownValue(knownValue) {}

  /// Return the value held by this lattice. This requires that the value is
  /// initialized.
  ValueT &getValue() {
    assert(!isUninitialized() && "expected known lattice element");
    return *optimisticValue;
  }
  const ValueT &getValue() const {
    return const_cast<Lattice<ValueT> *>(this)->getValue();
  }

  /// Returns true if the value of this lattice hasn't yet been initialized.
  bool isUninitialized() const override { return !optimisticValue.hasValue(); }
  /// Force the initialization of the element by setting it to its pessimistic
  /// fixpoint.
  ChangeResult defaultInitialize() override {
    return markPessimisticFixpoint();
  }

  /// Returns true if the lattice has reached a fixpoint. A fixpoint is when
  /// the information optimistically assumed to be true is the same as the
  /// information known to be true.
  bool isAtFixpoint() const override { return optimisticValue == knownValue; }

  /// Join the information contained in the 'rhs' lattice into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const AbstractSparseLattice &rhs) override {
    const Lattice<ValueT> &rhsLattice =
        static_cast<const Lattice<ValueT> &>(rhs);

    // If we are at a fixpoint, or rhs is uninitialized, there is nothing to do.
    if (isAtFixpoint() || rhsLattice.isUninitialized())
      return ChangeResult::NoChange;

    // Join the rhs value into this lattice.
    return join(rhsLattice.getValue());
  }

  /// Join the information contained in the 'rhs' value into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const ValueT &rhs) {
    // If the current lattice is uninitialized, copy the rhs value.
    if (isUninitialized()) {
      optimisticValue = rhs;
      return ChangeResult::Change;
    }

    // Otherwise, join rhs with the current optimistic value.
    ValueT newValue = ValueT::join(*optimisticValue, rhs);
    assert(ValueT::join(newValue, *optimisticValue) == newValue &&
           "expected `join` to be monotonic");
    assert(ValueT::join(newValue, rhs) == newValue &&
           "expected `join` to be monotonic");

    // Update the current optimistic value if something changed.
    if (newValue == optimisticValue)
      return ChangeResult::NoChange;

    optimisticValue = newValue;
    return ChangeResult::Change;
  }

  /// Mark the lattice element as having reached a pessimistic fixpoint. This
  /// means that the lattice may potentially have conflicting value states,
  /// and only the conservatively known value state should be relied on.
  ChangeResult markPessimisticFixpoint() override {
    if (isAtFixpoint())
      return ChangeResult::NoChange;

    // For this fixed point, we take whatever we knew to be true and set that
    // to our optimistic value.
    optimisticValue = knownValue;
    return ChangeResult::Change;
  }

  /// Print the lattice element.
  void print(raw_ostream &os) const override {
    os << "[";
    knownValue.print(os);
    os << ", ";
    if (optimisticValue) {
      optimisticValue->print(os);
    } else {
      os << "<NULL>";
    }
    os << "]";
  }

private:
  /// The value that is conservatively known to be true.
  ValueT knownValue;
  /// The currently computed value that is optimistically assumed to be true,
  /// or None if the lattice element is uninitialized.
  Optional<ValueT> optimisticValue;
};

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

/// This is a simple analysis state that represents whether the associated
/// program point (either a block or a control-flow edge) is live.
class Executable : public AnalysisState {
public:
  using AnalysisState::AnalysisState;

  /// The state is initialized by default.
  bool isUninitialized() const override { return false; }

  /// The state is always initialized.
  ChangeResult defaultInitialize() override { return ChangeResult::NoChange; }

  /// Set the state of the program point to live.
  ChangeResult setToLive();

  /// Get whether the program point is live.
  bool isLive() const { return live; }

  /// Print the liveness;
  void print(raw_ostream &os) const override;

  /// When the state of the program point is changed to live, re-invoke
  /// subscribed analyses on the operations in the block and on the block
  /// itself.
  void onUpdate(DataFlowSolver *solver) const override;

  /// Subscribe an analysis to changes to the liveness.
  void blockContentSubscribe(DataFlowAnalysis *analysis) {
    subscribers.insert(analysis);
  }

private:
  /// Whether the program point is live. Optimistically assume that the program
  /// point is dead.
  bool live = false;

  /// A set of analyses that should be updated when this state changes.
  SetVector<DataFlowAnalysis *, SmallVector<DataFlowAnalysis *, 4>,
            SmallPtrSet<DataFlowAnalysis *, 4>>
      subscribers;
};

//===----------------------------------------------------------------------===//
// ConstantValue
//===----------------------------------------------------------------------===//

/// This lattice value represents a known constant value of a lattice.
class ConstantValue {
public:
  /// Construct a constant value with a known constant.
  ConstantValue(Attribute knownValue = {}, Dialect *dialect = nullptr)
      : constant(knownValue), dialect(dialect) {}

  /// Get the constant value. Returns null if no value was determined.
  Attribute getConstantValue() const { return constant; }

  /// Get the dialect instance that can be used to materialize the constant.
  Dialect *getConstantDialect() const { return dialect; }

  /// Compare the constant values.
  bool operator==(const ConstantValue &rhs) const {
    return constant == rhs.constant;
  }

  /// The union with another constant value is null if they are different, and
  /// the same if they are the same.
  static ConstantValue join(const ConstantValue &lhs,
                            const ConstantValue &rhs) {
    return lhs == rhs ? lhs : ConstantValue();
  }

  /// Print the constant value.
  void print(raw_ostream &os) const;

private:
  /// The constant value.
  Attribute constant;
  /// An dialect instance that can be used to materialize the constant.
  Dialect *dialect;
};

//===----------------------------------------------------------------------===//
// PredecessorState
//===----------------------------------------------------------------------===//

/// This analysis state represents a set of known predecessors. This state is
/// used in sparse data-flow analysis to reason about region control-flow and
/// callgraphs. The state may also indicate that not all predecessors can be
/// known, if for example not all callsites of a callable are visible.
class PredecessorState : public AnalysisState {
public:
  using AnalysisState::AnalysisState;

  /// The state is initialized by default.
  bool isUninitialized() const override { return false; }

  /// The state is always initialized.
  ChangeResult defaultInitialize() override { return ChangeResult::NoChange; }

  /// Print the known predecessors.
  void print(raw_ostream &os) const override;

  /// Returns true if all predecessors are known.
  bool allPredecessorsKnown() const { return allKnown; }

  /// Indicate that there are potentially unknown predecessors.
  ChangeResult setHasUnknownPredecessors() {
    if (!allKnown)
      return ChangeResult::NoChange;
    allKnown = false;
    return ChangeResult::Change;
  }

  /// Get the known predecessors.
  ArrayRef<Operation *> getKnownPredecessors() const {
    return knownPredecessors.getArrayRef();
  }

  /// Add a known predecessor.
  ChangeResult join(Operation *predecessor) {
    return knownPredecessors.insert(predecessor) ? ChangeResult::Change
                                                 : ChangeResult::NoChange;
  }

private:
  /// Whether all predecessors are known. Optimistically assume that we know
  /// all predecessors.
  bool allKnown = true;

  /// The known control-flow predecessors of this program point.
  SetVector<Operation *, SmallVector<Operation *, 4>,
            SmallPtrSet<Operation *, 4>>
      knownPredecessors;
};

//===----------------------------------------------------------------------===//
// CFGEdge
//===----------------------------------------------------------------------===//

/// This program point represents a control-flow edge between a block and one
/// of its successors.
class CFGEdge
    : public GenericProgramPointBase<CFGEdge, std::pair<Block *, Block *>> {
public:
  using Base::Base;

  /// Get the block from which the edge originates.
  Block *getFrom() const { return getValue().first; }
  /// Get the target block.
  Block *getTo() const { return getValue().second; }

  /// Print the blocks between the control-flow edge.
  void print(raw_ostream &os) const override;
  /// Get a fused location of both blocks.
  Location getLoc() const override;
};

//===----------------------------------------------------------------------===//
// DeadCodeAnalysis
//===----------------------------------------------------------------------===//

/// Dead code analysis analyzes control-flow, as understood by
/// `RegionBranchOpInterface` and `BranchOpInterface`, and the callgraph, as
/// understood by `CallableOpInterface` and `CallOpInterface`.
///
/// This analysis uses known constant values of operands to determine the
/// liveness of each block and each edge between a block and its predecessors.
/// For region control-flow, this analysis determines the predecessor operations
/// for region entry blocks and region control-flow operations. For the
/// callgraph, this analysis determines the callsites and live returns of every
/// function.
class DeadCodeAnalysis : public DataFlowAnalysis {
public:
  explicit DeadCodeAnalysis(DataFlowSolver &solver);

  /// Initialize the analysis by visiting every operation with potential
  /// control-flow semantics.
  LogicalResult initialize(Operation *top) override;

  /// Visit an operation with control-flow semantics and deduce which of its
  /// successors are live.
  LogicalResult visit(ProgramPoint point) override;

private:
  /// Find and mark symbol callables with potentially unknown callsites as
  /// having overdefined predecessors. `top` is the top-level operation that the
  /// analysis is operating on.
  void initializeSymbolCallables(Operation *top);

  /// Recursively Initialize the analysis on nested regions.
  LogicalResult initializeRecursively(Operation *op);

  /// Visit the given call operation and compute any necessary lattice state.
  void visitCallOperation(CallOpInterface call);

  /// Visit the given branch operation with successors and try to determine
  /// which are live from the current block.
  void visitBranchOperation(BranchOpInterface branch);

  /// Visit the given region branch operation, which defines regions, and
  /// compute any necessary lattice state. This also resolves the lattice state
  /// of both the operation results and any nested regions.
  void visitRegionBranchOperation(RegionBranchOpInterface branch);

  /// Visit the given terminator operation that exits a region under an
  /// operation with control-flow semantics. These are terminators with no CFG
  /// successors.
  void visitRegionTerminator(Operation *op, RegionBranchOpInterface branch);

  /// Visit the given terminator operation that exits a callable region. These
  /// are terminators with no CFG successors.
  void visitCallableTerminator(Operation *op, CallableOpInterface callable);

  /// Mark the edge between `from` and `to` as executable.
  void markEdgeLive(Block *from, Block *to);

  /// Mark the entry blocks of the operation as executable.
  void markEntryBlocksLive(Operation *op);

  /// Get the constant values of the operands of the operation. Returns none if
  /// any of the operand lattices are uninitialized.
  Optional<SmallVector<Attribute>> getOperandValues(Operation *op);

  /// A symbol table used for O(1) symbol lookups during simplification.
  SymbolTableCollection symbolTable;
};

} // end namespace mlir

#endif // MLIR_ANALYSIS_SPARSEDATAFLOWANALYSIS_H
