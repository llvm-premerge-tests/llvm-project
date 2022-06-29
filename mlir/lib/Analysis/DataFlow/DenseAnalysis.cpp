//===- DenseAnalysis.cpp - Dense data-flow analysis -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DenseAnalysis.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// AbstractDenseDataFlowAnalysis
//===----------------------------------------------------------------------===//

LogicalResult AbstractDenseDataFlowAnalysis::initialize(Operation *top) {
  // Visit every operation and block.
  visitOperation(top);
  for (Region &region : top->getRegions()) {
    for (Block &block : region) {
      visitBlock(&block);
      for (Operation &op : block)
        if (failed(initialize(&op)))
          return failure();
    }
  }
  return success();
}

LogicalResult AbstractDenseDataFlowAnalysis::visit(ProgramPoint point) {
  if (auto *op = point.dyn_cast<Operation *>())
    visitOperation(op);
  else if (auto *block = point.dyn_cast<Block *>())
    visitBlock(block);
  else
    return failure();
  return success();
}

void AbstractDenseDataFlowAnalysis::visitOperation(Operation *op) {
  // If the containing block is not executable, bail out.
  if (!getOrCreateFor<Executable>(op, op->getBlock())->isLive())
    return;

  // Get the dense lattice to update.
  AbstractDenseElement *after = getLattice(op);
  if (after->get()->isAtFixpoint())
    return;

  // If this op implements region control-flow, then control-flow dictates its
  // transfer function.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op))
    return visitRegionBranchOperation(op, branch, after);

  // If this is a call operation, then join its lattices across known return
  // sites.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    const auto *predecessors = getOrCreateFor<PredecessorState>(op, call);
    // If not all return sites are known, then conservatively assume we can't
    // reason about the data-flow.
    if (!predecessors->allPredecessorsKnown())
      return markPessimisticFixpoint(after);
    return update(after, [this, predecessors, op](AbstractDenseState *state) {
      ChangeResult result = ChangeResult::NoChange;
      for (Operation *predecessor : predecessors->getKnownPredecessors())
        result |= state->join(*getLatticeFor(op, predecessor));
      return result;
    });
  }

  // Get the dense state before the execution of the op.
  const AbstractDenseState *before;
  if (Operation *prev = op->getPrevNode())
    before = getLatticeFor(op, prev);
  else
    before = getLatticeFor(op, op->getBlock());
  // If the incoming lattice is uninitialized, bail out.
  if (before->isUninitialized())
    return;

  // Invoke the operation transfer function.
  visitOperationImpl(op, *before, after);
}

void AbstractDenseDataFlowAnalysis::visitBlock(Block *block) {
  // If the block is not executable, bail out.
  if (!getOrCreateFor<Executable>(block, block)->isLive())
    return;

  // Get the dense lattice to update.
  AbstractDenseElement *after = getLattice(block);
  if (after->get()->isAtFixpoint())
    return;

  // The dense lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<PredecessorState>(block, callable);
      // If not all callsites are known, conservatively mark all lattices as
      // having reached their pessimistic fixpoints.
      if (!callsites->allPredecessorsKnown())
        return markPessimisticFixpoint(after);
      return update(after, [this, callsites, block](AbstractDenseState *state) {
        ChangeResult result = ChangeResult::NoChange;
        for (Operation *callsite : callsites->getKnownPredecessors()) {
          if (Operation *prev = callsite->getPrevNode())
            result |= state->join(*getLatticeFor(block, prev));
          else
            result |= state->join(*getLatticeFor(block, callsite->getBlock()));
        }
        return result;
      });
    }

    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp()))
      return visitRegionBranchOperation(block, branch, after);

    // Otherwise, we can't reason about the data-flow.
    return markPessimisticFixpoint(after);
  }

  // Join the state with the state after the block's predecessors.
  update(after, [this, block](AbstractDenseState *state) {
    ChangeResult result = ChangeResult::NoChange;
    for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end();
         it != e; ++it) {
      // Skip control edges that aren't executable.
      Block *predecessor = *it;
      if (!getOrCreateFor<Executable>(
               block, getProgramPoint<CFGEdge>(predecessor, block))
               ->isLive())
        continue;

      // Merge in the state from the predecessor's terminator.
      result |=
          state->join(*getLatticeFor(block, predecessor->getTerminator()));
    }
    return result;
  });
}

void AbstractDenseDataFlowAnalysis::visitRegionBranchOperation(
    ProgramPoint point, RegionBranchOpInterface branch,
    AbstractDenseElement *after) {
  // Get the terminator predecessors.
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  update(after, [&](AbstractDenseState *state) {
    ChangeResult result = ChangeResult::NoChange;
    for (Operation *op : predecessors->getKnownPredecessors()) {
      const AbstractDenseState *before;
      // If the predecessor is the parent, get the state before the parent.
      if (op == branch) {
        if (Operation *prev = op->getPrevNode())
          before = getLatticeFor(point, prev);
        else
          before = getLatticeFor(point, op->getBlock());

        // Otherwise, get the state after the terminator.
      } else {
        before = getLatticeFor(point, op);
      }
      result |= state->join(*before);
    }
    return result;
  });
}

const AbstractDenseState *
AbstractDenseDataFlowAnalysis::getLatticeFor(ProgramPoint dependent,
                                             ProgramPoint point) {
  AbstractDenseElement *element = getLattice(point);
  element->addDependency(this, dependent);
  return element->get();
}
