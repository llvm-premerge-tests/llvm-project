//===- SparseAnalysis.h - Sparse data-flow analysis -----------------------===//
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

#ifndef MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H
#define MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H

#include "mlir/Analysis/DataFlowFramework.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "llvm/ADT/SmallPtrSet.h"

namespace mlir {
namespace dataflow {

//===----------------------------------------------------------------------===//
// AbstractSparseState
//===----------------------------------------------------------------------===//

class AbstractSparseState : public AbstractState {
public:
  /// Join the information contained in 'rhs' into this state. Returns
  /// if the value of the state changed.
  virtual ChangeResult join(const AbstractSparseState &rhs) = 0;

  /// Returns true if the lattice state is at fixpoint and further calls to
  /// `join` will not update the value of the state.
  virtual bool isAtFixpoint() const = 0;

  /// Mark the lattice state as having reached a pessimistic fixpoint. This
  /// means that the lattice may potentially have an overdefined or underdefined
  /// value state, and only the most conservative value should be relied on.
  virtual ChangeResult markPessimisticFixpoint() = 0;

  /// Returns true if the value of this lattice hasn't yet been initialized.
  virtual bool isUninitialized() const = 0;
};

//===----------------------------------------------------------------------===//
// AbstractSparseElement
//===----------------------------------------------------------------------===//

class AbstractSparseElement : public AbstractElement {
public:
  /// Sparse elements can only be created on SSA values.
  explicit AbstractSparseElement(DataFlowSolver &solver, Value value)
      : AbstractElement(solver, value) {}

  virtual void useDefSubscribe(DataFlowAnalysis *analysis) = 0;

  virtual const AbstractSparseState *get() const override = 0;
};

/// This class represents a sparse analysis element. A sparse element is
/// attached to an SSA value and can track its dependents through the value's
/// use-def chain. This is useful for improving the performance of sparse
/// analyses where users are always dependents of SSA value elements.
template <typename StateT, template <typename, typename> class BaseT>
class SparseElement : public BaseT<StateT, AbstractSparseElement> {
public:
  using BaseT<StateT, AbstractSparseElement>::BaseT;

  /// When the sparse element gets updated, propagate an update to users of the
  /// value using its use-def chain to subscribed analyses.
  void onUpdate() override {
    for (Operation *user : this->point.template get<Value>().getUsers())
      for (DataFlowAnalysis *analysis : useDefSubscribers)
        this->solver.enqueue({user, analysis});
  }

  /// Subscribe an analysis to updates of the sparse element. When the element
  /// changes, subscribed analyses are re-invoked on all users of the value.
  /// This is more efficient than relying on the dependency map.
  void useDefSubscribe(DataFlowAnalysis *analysis) override {
    useDefSubscribers.insert(analysis);
  }

private:
  /// A set of analyses that should be updated when this element changes.
  SetVector<DataFlowAnalysis *, SmallVector<DataFlowAnalysis *, 4>,
            SmallPtrSet<DataFlowAnalysis *, 4>>
      useDefSubscribers;
};

//===----------------------------------------------------------------------===//
// OptimisticSparseState
//===----------------------------------------------------------------------===//

/// This class represents a sparse state that has an optimistic and known value.
/// This class should be used when the overdefined/underdefined value state is
/// not finitely representable.
template <typename ValueT>
class OptimisticSparseState : public AbstractSparseState {
public:
  template <typename PointT>
  explicit OptimisticSparseState(PointT point)
      : knownValue(ValueT::getPessimisticValue(point)) {}

  /// Return the value held by this lattice. This requires that the value is
  /// initialized.
  ValueT &getValue() {
    assert(!isUninitialized() && "expected known lattice element");
    return *optimisticValue;
  }
  const ValueT &getValue() const {
    return const_cast<OptimisticSparseState<ValueT> *>(this)->getValue();
  }

  /// Returns true if the value of this lattice hasn't yet been initialized.
  bool isUninitialized() const override { return !optimisticValue.hasValue(); }

  /// Returns true if the lattice has reached a fixpoint. A fixpoint is when
  /// the information optimistically assumed to be true is the same as the
  /// information known to be true.
  bool isAtFixpoint() const override { return optimisticValue == knownValue; }

  /// Join the information contained in the 'rhs' lattice into this
  /// lattice. Returns if the state of the current lattice changed.
  ChangeResult join(const AbstractSparseState &rhs) override {
    auto &rhsLattice = static_cast<const OptimisticSparseState<ValueT> &>(rhs);

    // If we are at a fixpoint, or rhs is uninitialized, there is nothing to do.
    if (isAtFixpoint() || rhsLattice.isUninitialized())
      return ChangeResult::NoChange;

    // Join the rhs value into this lattice.
    return join(rhsLattice.getValue());
  }

  ChangeResult meet(const OptimisticSparseState<ValueT> &rhs) {
    if (isUninitialized())
      return ChangeResult::NoChange;
    if (rhs.isUninitialized()) {
      optimisticValue.reset();
      return ChangeResult::Change;
    }
    ValueT newValue = ValueT::meet(getValue(), rhs.getValue());
    if (newValue == optimisticValue)
      return ChangeResult::NoChange;

    optimisticValue = newValue;
    return ChangeResult::Change;
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

  void print(raw_ostream &os) const override {
    os << '[';
    knownValue.print(os);
    if (optimisticValue) {
      os << ", ";
      optimisticValue->print(os);
    }
    os << ']';
  }

private:
  /// The value that is conservatively known to be true.
  ValueT knownValue;
  /// The currently computed value that is optimistically assumed to be true,
  /// or None if the lattice element is uninitialized.
  Optional<ValueT> optimisticValue;
};

//===----------------------------------------------------------------------===//
// AbstractSparseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for sparse (forward) data-flow analyses. A sparse analysis
/// implements a transfer function on operations from the lattices of the
/// operands to the lattices of the results. This analysis will propagate
/// lattices across control-flow edges and the callgraph using liveness
/// information.
class AbstractSparseDataFlowAnalysis : public DataFlowAnalysis {
public:
  /// Initialize the analysis by visiting every owner of an SSA value: all
  /// operations and blocks.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point. If this is a block and all control-flow
  /// predecessors or callsites are known, then the arguments lattices are
  /// propagated from them. If this is a call operation or an operation with
  /// region control-flow, then its result lattices are set accordingly.
  /// Otherwise, the operation transfer function is invoked.
  LogicalResult visit(ProgramPoint point) override;

protected:
  explicit AbstractSparseDataFlowAnalysis(DataFlowSolver &solver);

  /// The operation transfer function. Given the operand lattices, this
  /// function is expected to set the result lattices.
  virtual void visitOperationImpl(
      Operation *op,
      ArrayRef<const AbstractSparseState *> operandLattices,
      ArrayRef<AbstractSparseElement *> resultLattices) = 0;

  /// Given an operation with region control-flow, the lattices of the operands,
  /// and a region successor, compute the lattice values for block arguments
  /// that are not accounted for by the branching control flow (ex. the bounds
  /// of loops).
  virtual void visitNonControlFlowArgumentsImpl(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<AbstractSparseElement *> argLattices,
      unsigned firstIndex) = 0;

  /// Get the lattice element of a value.
  virtual AbstractSparseElement *getLatticeElement(Value value) = 0;

  /// Get a read-only lattice element for a value and add it as a dependency to
  /// a program point.
  const AbstractSparseState *getLatticeElementFor(ProgramPoint point,
                                                          Value value);

  /// Mark a lattice element as having reached its pessimistic fixpoint and
  /// propgate an update if changed.
  void markPessimisticFixpoint(AbstractSparseElement *element);

  /// Mark the given lattice elements as having reached their pessimistic
  /// fixpoints and propagate an update if any changed.
  void markAllPessimisticFixpoint(
      ArrayRef<AbstractSparseElement *> elements);

  /// Join the lattice element and propagate and update if it changed.
  void join(AbstractSparseElement *lhs,
            const AbstractSparseState &rhs);

private:
  /// Recursively initialize the analysis on nested operations and blocks.
  LogicalResult initializeRecursively(Operation *op);

  /// Visit an operation. If this is a call operation or an operation with
  /// region control-flow, then its result lattices are set accordingly.
  /// Otherwise, the operation transfer function is invoked.
  void visitOperation(Operation *op);

  /// If this is a block and all control-flow predecessors or callsites are
  /// known, then the arguments lattices are propagated from them.
  void visitBlock(Block *block);

  /// Visit a program point `point` with predecessors within a region branch
  /// operation `branch`, which can either be the entry block of one of the
  /// regions or the parent operation itself, and set either the argument or
  /// parent result lattices.
  void
  visitRegionSuccessors(ProgramPoint point, RegionBranchOpInterface branch,
                        Optional<unsigned> successorIndex,
                        ArrayRef<AbstractSparseElement *> elements);
};

//===----------------------------------------------------------------------===//
// SparseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A sparse (forward) data-flow analysis for propagating SSA value lattices
/// across the IR by implementing transfer functions for operations.
///
/// `StateT` is expected to be a subclass of `AbstractSparseState`.
template <typename StateT>
class SparseDataFlowAnalysis : public AbstractSparseDataFlowAnalysis {
public:
  explicit SparseDataFlowAnalysis(DataFlowSolver &solver)
      : AbstractSparseDataFlowAnalysis(solver) {}

  /// Visit an operation with the lattices of its operands. This function is
  /// expected to set the lattices of the operation's results.
  virtual void
  visitOperation(Operation *op, ArrayRef<const StateT *> operands,
                 ArrayRef<typename StateT::ElementT *> results) = 0;

  /// Given an operation with possible region control-flow, the lattices of the
  /// operands, and a region successor, compute the lattice values for block
  /// arguments that are not accounted for by the branching control flow (ex.
  /// the bounds of loops). By default, this method marks all such lattice
  /// elements as having reached a pessimistic fixpoint. `firstIndex` is the
  /// index of the first element of `argLattices` that is set by control-flow.
  virtual void visitNonControlFlowArguments(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<typename StateT::ElementT *> argLattices, unsigned firstIndex) {
    markAllPessimisticFixpoint(argLattices.take_front(firstIndex));
    markAllPessimisticFixpoint(argLattices.drop_front(
        firstIndex + successor.getSuccessorInputs().size()));
  }

protected:
  bool staticallyProvides(TypeID stateID, ProgramPoint point) const override {
    return stateID == TypeID::get<StateT>() && point.is<Value>();
  }

  /// Get the lattice element for a value.
  typename StateT::ElementT *getLatticeElement(Value value) override {
    return getOrCreate<StateT>(value);
  }

  /// Get the lattice element for a value and create a dependency on the
  /// provided program point.
  const StateT *getLatticeElementFor(ProgramPoint point, Value value) {
    return static_cast<const StateT *>(
        AbstractSparseDataFlowAnalysis::getLatticeElementFor(point, value));
  }

  /// Mark the lattice elements of a range of values as having reached their
  /// pessimistic fixpoint.
  void
  markAllPessimisticFixpoint(ArrayRef<typename StateT::ElementT *> elements) {
    AbstractSparseDataFlowAnalysis::markAllPessimisticFixpoint(
        {reinterpret_cast<AbstractSparseElement *const *>(
             elements.begin()),
         elements.size()});
  }

private:
  /// Type-erased wrappers that convert the abstract lattice operands to derived
  /// lattices and invoke the virtual hooks operating on the derived lattices.
  void visitOperationImpl(
      Operation *op,
      ArrayRef<const AbstractSparseState *> operandLattices,
      ArrayRef<AbstractSparseElement *> resultLattices) override {
    visitOperation(
        op,
        {reinterpret_cast<const StateT *const *>(operandLattices.begin()),
         operandLattices.size()},
        {reinterpret_cast<typename StateT::ElementT *const *>(
             resultLattices.begin()),
         resultLattices.size()});
  }
  void visitNonControlFlowArgumentsImpl(
      Operation *op, const RegionSuccessor &successor,
      ArrayRef<AbstractSparseElement *> argLattices,
      unsigned firstIndex) override {
    visitNonControlFlowArguments(
        op, successor,
        {reinterpret_cast<typename StateT::ElementT *const *>(
             argLattices.begin()),
         argLattices.size()},
        firstIndex);
  }
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DATAFLOW_SPARSEANALYSIS_H
