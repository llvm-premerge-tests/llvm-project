//===- SparseAnalysis.cpp - Sparse data-flow analysis ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/SparseAnalysis.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// AbstractSparseDataFlowAnalysis
//===----------------------------------------------------------------------===//

AbstractSparseDataFlowAnalysis::AbstractSparseDataFlowAnalysis(
    DataFlowSolver &solver)
    : DataFlowAnalysis(solver) {
  registerPointKind<CFGEdge>();
}

LogicalResult AbstractSparseDataFlowAnalysis::initialize(Operation *top) {
  // Mark the entry block arguments as having reached their pessimistic
  // fixpoints.
  for (Region &region : top->getRegions()) {
    if (region.empty())
      continue;
    for (Value argument : region.front().getArguments())
      markPessimisticFixpoint(getLatticeElement(argument));
  }

  return initializeRecursively(top);
}

LogicalResult
AbstractSparseDataFlowAnalysis::initializeRecursively(Operation *op) {
  // Initialize the analysis by visiting every owner of an SSA value (all
  // operations and blocks).
  visitOperation(op);
  for (Region &region : op->getRegions()) {
    for (Block &block : region) {
      getOrCreate<Executable>(&block)->blockContentSubscribe(this);
      visitBlock(&block);
      for (Operation &op : block)
        if (failed(initializeRecursively(&op)))
          return failure();
    }
  }

  return success();
}

LogicalResult AbstractSparseDataFlowAnalysis::visit(ProgramPoint point) {
  if (Operation *op = point.dyn_cast<Operation *>())
    visitOperation(op);
  else if (Block *block = point.dyn_cast<Block *>())
    visitBlock(block);
  else
    return failure();
  return success();
}

void AbstractSparseDataFlowAnalysis::visitOperation(Operation *op) {
  // Exit early on operations with no results.
  if (op->getNumResults() == 0)
    return;

  // If the containing block is not executable, bail out.
  if (!getOrCreate<Executable>(op->getBlock())->get()->isLive())
    return;

  // Get the result lattices.
  SmallVector<AbstractSparseElement *> resultElements;
  resultElements.reserve(op->getNumResults());
  // Track whether all results have reached their fixpoint.
  bool allAtFixpoint = true;
  for (Value result : op->getResults()) {
    AbstractSparseElement *resultElement = getLatticeElement(result);
    allAtFixpoint &= resultElement->get()->isAtFixpoint();
    resultElements.push_back(resultElement);
  }
  // If all result lattices have reached a fixpoint, there is nothing to do.
  if (allAtFixpoint)
    return;

  // The results of a region branch operation are determined by control-flow.
  if (auto branch = dyn_cast<RegionBranchOpInterface>(op))
    return visitRegionSuccessors({branch}, branch, llvm::None, resultElements);

  // The results of a call operation are determined by the callgraph.
  if (auto call = dyn_cast<CallOpInterface>(op)) {
    const auto *predecessors = getOrCreateFor<PredecessorState>(op, call);
    // If not all return sites are known, then conservatively assume we can't
    // reason about the data-flow.
    if (!predecessors->allPredecessorsKnown())
      return markAllPessimisticFixpoint(resultElements);
    for (Operation *predecessor : predecessors->getKnownPredecessors())
      for (auto it : llvm::zip(predecessor->getOperands(), resultElements))
        join(std::get<1>(it), *getLatticeElementFor(op, std::get<0>(it)));
    return;
  }

  // Grab the lattice elements of the operands.
  SmallVector<const AbstractSparseState *> operandStates;
  operandStates.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    AbstractSparseElement *operandElement = getLatticeElement(operand);
    operandElement->useDefSubscribe(this);
    // If any of the operand states are not initialized, bail out.
    if (operandElement->get()->isUninitialized())
      return;
    operandStates.push_back(operandElement->get());
  }

  // Invoke the operation transfer function.
  visitOperationImpl(op, operandStates, resultElements);
}

void AbstractSparseDataFlowAnalysis::visitBlock(Block *block) {
  // Exit early on blocks with no arguments.
  if (block->getNumArguments() == 0)
    return;

  // If the block is not executable, bail out.
  if (!getOrCreate<Executable>(block)->get()->isLive())
    return;

  // Get the argument lattices.
  SmallVector<AbstractSparseElement *> argElements;
  argElements.reserve(block->getNumArguments());
  bool allAtFixpoint = true;
  for (BlockArgument argument : block->getArguments()) {
    AbstractSparseElement *argElement = getLatticeElement(argument);
    allAtFixpoint &= argElement->get()->isAtFixpoint();
    argElements.push_back(argElement);
  }
  // If all argument lattices have reached their fixpoints, then there is
  // nothing to do.
  if (allAtFixpoint)
    return;

  // The argument lattices of entry blocks are set by region control-flow or the
  // callgraph.
  if (block->isEntryBlock()) {
    // Check if this block is the entry block of a callable region.
    auto callable = dyn_cast<CallableOpInterface>(block->getParentOp());
    if (callable && callable.getCallableRegion() == block->getParent()) {
      const auto *callsites = getOrCreateFor<PredecessorState>(block, callable);
      // If not all callsites are known, conservatively mark all lattices as
      // having reached their pessimistic fixpoints.
      if (!callsites->allPredecessorsKnown())
        return markAllPessimisticFixpoint(argElements);
      for (Operation *callsite : callsites->getKnownPredecessors()) {
        auto call = cast<CallOpInterface>(callsite);
        for (auto it : llvm::zip(call.getArgOperands(), argElements))
          join(std::get<1>(it), *getLatticeElementFor(block, std::get<0>(it)));
      }
      return;
    }

    // Check if the lattices can be determined from region control flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(block->getParentOp())) {
      return visitRegionSuccessors(
          block, branch, block->getParent()->getRegionNumber(), argElements);
    }

    // Otherwise, we can't reason about the data-flow.
    return visitNonControlFlowArgumentsImpl(block->getParentOp(),
                                            RegionSuccessor(block->getParent()),
                                            argElements, /*firstIndex=*/0);
  }

  // Iterate over the predecessors of the non-entry block.
  for (Block::pred_iterator it = block->pred_begin(), e = block->pred_end();
       it != e; ++it) {
    Block *predecessor = *it;

    // If the edge from the predecessor block to the current block is not live,
    // bail out.
    auto *edgeExecutable =
        getOrCreate<Executable>(getProgramPoint<CFGEdge>(predecessor, block));
    edgeExecutable->blockContentSubscribe(this);
    if (!edgeExecutable->get()->isLive())
      continue;

    // Check if we can reason about the data-flow from the predecessor.
    if (auto branch =
            dyn_cast<BranchOpInterface>(predecessor->getTerminator())) {
      SuccessorOperands operands =
          branch.getSuccessorOperands(it.getSuccessorIndex());
      for (auto &it : llvm::enumerate(argElements)) {
        if (Value operand = operands[it.index()]) {
          join(it.value(), *getLatticeElementFor(block, operand));
        } else {
          // Conservatively mark internally produced arguments as having reached
          // their pessimistic fixpoint.
          markPessimisticFixpoint(it.value());
        }
      }
    } else {
      return markAllPessimisticFixpoint(argElements);
    }
  }
}

void AbstractSparseDataFlowAnalysis::visitRegionSuccessors(
    ProgramPoint point, RegionBranchOpInterface branch,
    Optional<unsigned> successorIndex,
    ArrayRef<AbstractSparseElement *> elements) {
  const auto *predecessors = getOrCreateFor<PredecessorState>(point, point);
  assert(predecessors->allPredecessorsKnown() &&
         "unexpected unresolved region successors");

  for (Operation *op : predecessors->getKnownPredecessors()) {
    // Get the incoming successor operands.
    Optional<OperandRange> operands;

    // Check if the predecessor is the parent op.
    if (op == branch) {
      operands = branch.getSuccessorEntryOperands(successorIndex);
      // Otherwise, try to deduce the operands from a region return-like op.
    } else {
      if (isRegionReturnLike(op))
        operands = getRegionBranchSuccessorOperands(op, successorIndex);
    }

    if (!operands) {
      // We can't reason about the data-flow.
      return markAllPessimisticFixpoint(elements);
    }

    ValueRange inputs = predecessors->getSuccessorInputs(op);
    assert(inputs.size() == operands->size() &&
           "expected the same number of successor inputs as operands");

    unsigned firstIndex = 0;
    if (inputs.size() != elements.size()) {
      if (auto *op = point.dyn_cast<Operation *>()) {
        if (!inputs.empty())
          firstIndex = inputs.front().cast<OpResult>().getResultNumber();
        visitNonControlFlowArgumentsImpl(
            branch,
            RegionSuccessor(
                branch->getResults().slice(firstIndex, inputs.size())),
            elements, firstIndex);
      } else {
        if (!inputs.empty())
          firstIndex = inputs.front().cast<BlockArgument>().getArgNumber();
        Region *region = point.get<Block *>()->getParent();
        visitNonControlFlowArgumentsImpl(
            branch,
            RegionSuccessor(region, region->getArguments().slice(
                                        firstIndex, inputs.size())),
            elements, firstIndex);
      }
    }

    for (auto it : llvm::zip(*operands, elements.drop_front(firstIndex)))
      join(std::get<1>(it), *getLatticeElementFor(point, std::get<0>(it)));
  }
}

const AbstractSparseState *
AbstractSparseDataFlowAnalysis::getLatticeElementFor(ProgramPoint point,
                                                     Value value) {
  AbstractSparseElement *element = getLatticeElement(value);
  element->addDependency(this, point);
  return element->get();
}

void AbstractSparseDataFlowAnalysis::markPessimisticFixpoint(
    AbstractSparseElement *element) {
  element->update(this, [](AbstractState *state) {
    return static_cast<AbstractSparseState *>(state)
        ->markPessimisticFixpoint();
  });
}

void AbstractSparseDataFlowAnalysis::markAllPessimisticFixpoint(
    ArrayRef<AbstractSparseElement *> elements) {
  for (AbstractSparseElement *element : elements)
    markPessimisticFixpoint(element);
}

void AbstractSparseDataFlowAnalysis::join(
    AbstractSparseElement *lhs,
    const AbstractSparseState &rhs) {
  lhs->update(this, [&rhs](AbstractState *lhsState) {
    return static_cast<AbstractSparseState *>(lhsState)->join(rhs);
  });
}

