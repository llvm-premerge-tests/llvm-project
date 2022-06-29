//===- DenseAnalysis.h - Dense data-flow analysis -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements dense data-flow analysis using the data-flow analysis
// framework. The analysis is forward and conditional and uses the results of
// dead code analysis to prune dead code during the analysis.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H
#define MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/SparseAnalysis.h"

namespace mlir {
namespace dataflow {

//===----------------------------------------------------------------------===//
// AbstractDenseState
//===----------------------------------------------------------------------===//

/// This class represents a dense lattice. A dense lattice is attached to
/// operations to represent the program state after their execution or to blocks
/// to represent the program state at the beginning of the block. A dense
/// lattice is propagated through the IR by dense data-flow analysis.
using AbstractDenseState = AbstractSparseState;

class AbstractDenseElement : public AbstractElement {
public:
  using AbstractElement::AbstractElement;

  virtual const AbstractDenseState *get() const override = 0;
};

//===----------------------------------------------------------------------===//
// AbstractDenseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// Base class for dense data-flow analyses. Dense data-flow analysis attaches a
/// lattice between the execution of operations and implements a transfer
/// function from the lattice before each operation to the lattice after. The
/// lattice contains information about the state of the program at that point.
///
/// In this implementation, a lattice attached to an operation represents the
/// state of the program after its execution, and a lattice attached to block
/// represents the state of the program right before it starts executing its
/// body.
class AbstractDenseDataFlowAnalysis : public DataFlowAnalysis {
public:
  using DataFlowAnalysis::DataFlowAnalysis;

  /// Initialize the analysis by visiting every program point whose execution
  /// may modify the program state; that is, every operation and block.
  LogicalResult initialize(Operation *top) override;

  /// Visit a program point that modifies the state of the program. If this is a
  /// block, then the state is propagated from control-flow predecessors or
  /// callsites. If this is a call operation or region control-flow operation,
  /// then the state after the execution of the operation is set by control-flow
  /// or the callgraph. Otherwise, this function invokes the operation transfer
  /// function.
  LogicalResult visit(ProgramPoint point) override;

protected:
  /// Propagate the dense lattice before the execution of an operation to the
  /// lattice after its execution.
  virtual void visitOperationImpl(Operation *op,
                                  const AbstractDenseState &before,
                                  AbstractDenseElement *after) = 0;

  /// Get the dense element after the execution of the given program point.
  virtual AbstractDenseElement *getLattice(ProgramPoint point) = 0;

  /// Get the dense lattice after the execution of the given program point and
  /// add it as a dependency to a program point.
  const AbstractDenseState *getLatticeFor(ProgramPoint dependent,
                                          ProgramPoint point);

  void update(AbstractDenseElement *element,
              function_ref<ChangeResult(AbstractDenseState *)> updateFn) {
    element->update(this, [updateFn](AbstractState *state) {
      return updateFn(static_cast<AbstractDenseState *>(state));
    });
  }

  void markPessimisticFixpoint(AbstractDenseElement *element) {
    element->update(this, [](AbstractState *state) {
      return static_cast<AbstractDenseState *>(state)
          ->markPessimisticFixpoint();
    });
  }

private:
  /// Visit an operation. If this is a call operation or region control-flow
  /// operation, then the state after the execution of the operation is set by
  /// control-flow or the callgraph. Otherwise, this function invokes the
  /// operation transfer function.
  void visitOperation(Operation *op);

  /// Visit a block. The state at the start of the block is propagated from
  /// control-flow predecessors or callsites
  void visitBlock(Block *block);

  /// Visit a program point within a region branch operation with predecessors
  /// in it. This can either be an entry block of one of the regions of the
  /// parent operation itself.
  void visitRegionBranchOperation(ProgramPoint point,
                                  RegionBranchOpInterface branch,
                                  AbstractDenseElement *after);
};

//===----------------------------------------------------------------------===//
// DenseDataFlowAnalysis
//===----------------------------------------------------------------------===//

/// A dense (forward) data-flow analysis for propagating lattices before and
/// after the execution of every operation across the IR by implementing
/// transfer functions for operations.
///
/// `StateT` is expected to be a subclass of `AbstractDenseLattice`.
template <typename StateT>
class DenseDataFlowAnalysis : public AbstractDenseDataFlowAnalysis {
public:
  using AbstractDenseDataFlowAnalysis::AbstractDenseDataFlowAnalysis;

  /// Visit an operation with the dense lattice before its execution. This
  /// function is expected to set the dense lattice after its execution.
  virtual void visitOperation(Operation *op, const StateT &before,
                              typename StateT::ElementT *after) = 0;

protected:
  /// Get the dense lattice after this program point.
  typename StateT::ElementT *getLattice(ProgramPoint point) override {
    return getOrCreate<StateT>(point);
  }

private:
  /// Type-erased wrappers that convert the abstract dense lattice to a derived
  /// lattice and invoke the virtual hooks operating on the derived lattice.
  void visitOperationImpl(Operation *op, const AbstractDenseState &before,
                          AbstractDenseElement *after) override {
    visitOperation(op, static_cast<const StateT &>(before),
                   static_cast<typename StateT::ElementT *>(after));
  }
};

} // end namespace dataflow
} // end namespace mlir

#endif // MLIR_ANALYSIS_DENSEDATAFLOWANALYSIS_H
