//===- DeadCodeAnalysis.cpp - Dead code analysis --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/ConstantPropagationAnalysis.h"

using namespace mlir;
using namespace mlir::dataflow;

//===----------------------------------------------------------------------===//
// Executable
//===----------------------------------------------------------------------===//

ChangeResult Executable::setToLive() {
  if (live)
    return ChangeResult::NoChange;
  live = true;
  return ChangeResult::Change;
}

void Executable::print(raw_ostream &os) const {
  os << (live ? "live" : "dead");
}

//===----------------------------------------------------------------------===//
// PredecessorState
//===----------------------------------------------------------------------===//

void PredecessorState::print(raw_ostream &os) const {
  if (allPredecessorsKnown())
    os << "(all) ";
  os << "predecessors:\n";
  for (Operation *op : getKnownPredecessors())
    os << "  " << *op << "\n";
}

ChangeResult PredecessorState::join(Operation *predecessor) {
  return knownPredecessors.insert(predecessor) ? ChangeResult::Change
                                               : ChangeResult::NoChange;
}

ChangeResult PredecessorState::join(Operation *predecessor, ValueRange inputs) {
  ChangeResult result = join(predecessor);
  if (!inputs.empty()) {
    ValueRange &curInputs = successorInputs[predecessor];
    if (curInputs != inputs) {
      curInputs = inputs;
      result |= ChangeResult::Change;
    }
  }
  return result;
}

//===----------------------------------------------------------------------===//
// CFGEdge
//===----------------------------------------------------------------------===//

Location CFGEdge::getLoc() const {
  return FusedLoc::get(
      getFrom()->getParent()->getContext(),
      {getFrom()->getParent()->getLoc(), getTo()->getParent()->getLoc()});
}

void CFGEdge::print(raw_ostream &os) const {
  getFrom()->print(os);
  os << "\n -> \n";
  getTo()->print(os);
}

//===----------------------------------------------------------------------===//
// DeadCodeAnalysis
//===----------------------------------------------------------------------===//

DeadCodeAnalysis::DeadCodeAnalysis(DataFlowSolver &solver)
    : DataFlowAnalysis(solver) {
  registerPointKind<CFGEdge>();
}

LogicalResult DeadCodeAnalysis::initialize(Operation *top) {
  // Mark the top-level blocks as executable.
  for (Region &region : top->getRegions()) {
    if (region.empty())
      continue;
    update<Executable>(&region.front(),
                       [](Executable *state) { return state->setToLive(); });
  }

  // Mark as overdefined the predecessors of symbol callables with potentially
  // unknown predecessors.
  initializeSymbolCallables(top);

  return initializeRecursively(top);
}

void DeadCodeAnalysis::initializeSymbolCallables(Operation *top) {
  auto walkFn = [&](Operation *symTable, bool allUsesVisible) {
    Region &symbolTableRegion = symTable->getRegion(0);
    Block *symbolTableBlock = &symbolTableRegion.front();

    bool foundSymbolCallable = false;
    for (auto callable : symbolTableBlock->getOps<CallableOpInterface>()) {
      Region *callableRegion = callable.getCallableRegion();
      if (!callableRegion)
        continue;
      auto symbol = dyn_cast<SymbolOpInterface>(callable.getOperation());
      if (!symbol)
        continue;

      // Public symbol callables or those for which we can't see all uses have
      // potentially unknown callsites.
      if (symbol.isPublic() || (!allUsesVisible && symbol.isNested())) {
        update<PredecessorState>(callable, [](PredecessorState *state) {
          return state->setHasUnknownPredecessors();
        });
      }
      foundSymbolCallable = true;
    }

    // Exit early if no eligible symbol callables were found in the table.
    if (!foundSymbolCallable)
      return;

    // Walk the symbol table to check for non-call uses of symbols.
    Optional<SymbolTable::UseRange> uses =
        SymbolTable::getSymbolUses(&symbolTableRegion);
    if (!uses) {
      // If we couldn't gather the symbol uses, conservatively assume that
      // we can't track information for any nested symbols.
      return top->walk([&](CallableOpInterface callable) {
        update<PredecessorState>(callable, [](PredecessorState *state) {
          return state->setHasUnknownPredecessors();
        });
      });
    }

    for (const SymbolTable::SymbolUse &use : *uses) {
      if (isa<CallOpInterface>(use.getUser()))
        continue;
      // If a callable symbol has a non-call use, then we can't be guaranteed to
      // know all callsites.
      Operation *symbol = symbolTable.lookupSymbolIn(top, use.getSymbolRef());
      update<PredecessorState>(symbol, [](PredecessorState *state) {
        return state->setHasUnknownPredecessors();
      });
    }
  };
  SymbolTable::walkSymbolTables(top, !top->getBlock(), walkFn);
}

/// Returns true if the operation terminates a block. It is insufficient to
/// check for `OpTrait::IsTerminator` because unregistered operations can be
/// terminators.
static bool isTerminator(Operation *op) {
  if (op->hasTrait<OpTrait::IsTerminator>())
    return true;
  return &op->getBlock()->back() == op;
}

LogicalResult DeadCodeAnalysis::initializeRecursively(Operation *op) {
  // Initialize the analysis by visiting every op with control-flow semantics.
  if (op->getNumRegions() || op->getNumSuccessors() || isTerminator(op) ||
      isa<CallOpInterface>(op)) {
    // When the liveness of the parent block changes, make sure to re-invoke the
    // analysis on the op.
    if (op->getBlock())
      getOrCreate<Executable>(op->getBlock())->blockContentSubscribe(this);
    // Visit the op.
    if (failed(visit(op)))
      return failure();
  }
  // Recurse on nested operations.
  for (Region &region : op->getRegions())
    for (Operation &op : region.getOps())
      if (failed(initializeRecursively(&op)))
        return failure();
  return success();
}

void DeadCodeAnalysis::markEdgeLive(Block *from, Block *to) {
  update<Executable>(to, [](Executable *state) { return state->setToLive(); });
  update<Executable>(getProgramPoint<CFGEdge>(from, to),
                     [](Executable *state) { return state->setToLive(); });
}

void DeadCodeAnalysis::markEntryBlocksLive(Operation *op) {
  for (Region &region : op->getRegions()) {
    if (region.empty())
      continue;
    update<Executable>(&region.front(),
                       [](Executable *state) { return state->setToLive(); });
  }
}

LogicalResult DeadCodeAnalysis::visit(ProgramPoint point) {
  if (point.is<Block *>())
    return success();
  auto *op = point.dyn_cast<Operation *>();
  if (!op)
    return emitError(point.getLoc(), "unknown program point kind");

  // If the parent block is not executable, there is nothing to do.
  if (!getOrCreate<Executable>(op->getBlock())->get()->isLive())
    return success();

  // We have a live call op. Add this as a live predecessor of the callee.
  if (auto call = dyn_cast<CallOpInterface>(op))
    visitCallOperation(call);

  // Visit the regions.
  if (op->getNumRegions()) {
    // Check if we can reason about the region control-flow.
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op)) {
      visitRegionBranchOperation(branch);

      // Check if this is a callable operation.
    } else if (auto callable = dyn_cast<CallableOpInterface>(op)) {
      const auto *callsites = getOrCreateFor<PredecessorState>(op, callable);

      // If the callsites could not be resolved or are known to be non-empty,
      // mark the callable as executable.
      if (!callsites->allPredecessorsKnown() ||
          !callsites->getKnownPredecessors().empty())
        markEntryBlocksLive(callable);

      // Otherwise, conservatively mark all entry blocks as executable.
    } else {
      markEntryBlocksLive(op);
    }
  }

  if (isTerminator(op) && !op->getNumSuccessors()) {
    if (auto branch = dyn_cast<RegionBranchOpInterface>(op->getParentOp())) {
      // Visit the exiting terminator of a region.
      visitRegionTerminator(op, branch);
    } else if (auto callable =
                   dyn_cast<CallableOpInterface>(op->getParentOp())) {
      // Visit the exiting terminator of a callable.
      visitCallableTerminator(op, callable);
    }
  }
  // Visit the successors.
  if (op->getNumSuccessors()) {
    // Check if we can reason about the control-flow.
    if (auto branch = dyn_cast<BranchOpInterface>(op)) {
      visitBranchOperation(branch);

      // Otherwise, conservatively mark all successors as exectuable.
    } else {
      for (Block *successor : op->getSuccessors())
        markEdgeLive(op->getBlock(), successor);
    }
  }

  return success();
}

void DeadCodeAnalysis::visitCallOperation(CallOpInterface call) {
  Operation *callableOp = nullptr;
  if (Value callableValue = call.getCallableForCallee().dyn_cast<Value>())
    callableOp = callableValue.getDefiningOp();
  else
    callableOp = call.resolveCallable(&symbolTable);

  // A call to a externally-defined callable has unknown predecessors.
  const auto isExternalCallable = [](Operation *op) {
    if (auto callable = dyn_cast<CallableOpInterface>(op))
      return !callable.getCallableRegion();
    return false;
  };

  // TODO: Add support for non-symbol callables when necessary. If the
  // callable has non-call uses we would mark as having reached pessimistic
  // fixpoint, otherwise allow for propagating the return values out.
  if (isa_and_nonnull<SymbolOpInterface>(callableOp) &&
      !isExternalCallable(callableOp)) {
    // Add the live callsite.
    update<PredecessorState>(callableOp, [call](PredecessorState *state) {
      return state->join(call);
    });
  } else {
    // Mark this call op's predecessors as overdefined.
    update<PredecessorState>(call, [](PredecessorState *state) {
      return state->setHasUnknownPredecessors();
    });
  }
}

/// Get the constant values of the operands of an operation. If any of the
/// constant value lattices are uninitialized, return none to indicate the
/// analysis should bail out.
static Optional<SmallVector<Attribute>>
getOperandValuesImpl(Operation *op,
                     function_ref<const ConstantValueState *(Value)> getState) {
  SmallVector<Attribute> operands;
  operands.reserve(op->getNumOperands());
  for (Value operand : op->getOperands()) {
    const ConstantValueState *cv = getState(operand);
    // If any of the operands' values are uninitialized, bail out.
    if (cv->isUninitialized())
      return {};
    operands.push_back(cv->getValue().getConstantValue());
  }
  return operands;
}

Optional<SmallVector<Attribute>>
DeadCodeAnalysis::getOperandValues(Operation *op) {
  return getOperandValuesImpl(
      op, [&](Value value) -> const ConstantValueState * {
        auto *element = getOrCreate<ConstantValueState>(value);
        element->useDefSubscribe(this);
        return element->get();
      });
}

void DeadCodeAnalysis::visitBranchOperation(BranchOpInterface branch) {
  // Try to deduce a single successor for the branch.
  Optional<SmallVector<Attribute>> operands = getOperandValues(branch);
  if (!operands)
    return;

  if (Block *successor = branch.getSuccessorForOperands(*operands)) {
    markEdgeLive(branch->getBlock(), successor);
  } else {
    // Otherwise, mark all successors as executable and outgoing edges.
    for (Block *successor : branch->getSuccessors())
      markEdgeLive(branch->getBlock(), successor);
  }
}

void DeadCodeAnalysis::visitRegionBranchOperation(
    RegionBranchOpInterface branch) {
  // Try to deduce which regions are executable.
  Optional<SmallVector<Attribute>> operands = getOperandValues(branch);
  if (!operands)
    return;

  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(/*index=*/{}, *operands, successors);
  for (const RegionSuccessor &successor : successors) {
    // The successor can be either an entry block or the parent operation.
    ProgramPoint point = successor.getSuccessor()
                             ? &successor.getSuccessor()->front()
                             : ProgramPoint(branch);
    // Mark the entry block as executable.
    update<Executable>(point,
                       [](Executable *state) { return state->setToLive(); });
    // Add the parent op as a predecessor.
    update<PredecessorState>(
        point, [branch, &successor](PredecessorState *state) {
          return state->join(branch, successor.getSuccessorInputs());
        });
  }
}

void DeadCodeAnalysis::visitRegionTerminator(Operation *op,
                                             RegionBranchOpInterface branch) {
  Optional<SmallVector<Attribute>> operands = getOperandValues(op);
  if (!operands)
    return;

  SmallVector<RegionSuccessor> successors;
  branch.getSuccessorRegions(op->getParentRegion()->getRegionNumber(),
                             *operands, successors);

  // Mark successor region entry blocks as executable and add this op to the
  // list of predecessors.
  for (const RegionSuccessor &successor : successors) {
    if (Region *region = successor.getSuccessor()) {
      update<Executable>(&region->front(),
                         [](Executable *state) { return state->setToLive(); });
      update<PredecessorState>(
          &region->front(), [op, &successor](PredecessorState *state) {
            return state->join(op, successor.getSuccessorInputs());
          });
    } else {
      // Add this terminator as a predecessor to the parent op.
      update<PredecessorState>(
          branch, [op, &successor](PredecessorState *state) {
            return state->join(op, successor.getSuccessorInputs());
          });
    }
  }
}

void DeadCodeAnalysis::visitCallableTerminator(Operation *op,
                                               CallableOpInterface callable) {
  // If there are no exiting values, we have nothing to do.
  if (op->getNumOperands() == 0)
    return;

  // Add as predecessors to all callsites this return op.
  auto *callsites = getOrCreateFor<PredecessorState>(op, callable);
  bool canResolve = op->hasTrait<OpTrait::ReturnLike>();
  for (Operation *predecessor : callsites->getKnownPredecessors()) {
    assert(isa<CallOpInterface>(predecessor));
    auto *predecessors = getOrCreate<PredecessorState>(predecessor);
    if (canResolve) {
      predecessors->update(
          this, [op](PredecessorState *state) { return state->join(op); });
    } else {
      // If the terminator is not a return-like, then conservatively assume we
      // can't resolve the predecessor.
      predecessors->update(this, [](PredecessorState *state) {
        return state->setHasUnknownPredecessors();
      });
    }
  }
}
