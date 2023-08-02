//===- CFGToSCF.h - Control Flow Graph to Structured Control Flow *- C++ -*===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This code is an implementation of:
// Helge Bahmann, Nico Reissmann, Magnus Jahre, and Jan Christian Meyer. 2015.
// Perfect Reconstructability of Control Flow from Demand Dependence Graphs. ACM
// Trans. Archit. Code Optim. 11, 4, Article 66 (January 2015), 25 pages.
// https://doi.org/10.1145/2693261
//
// It defines an algorithm to translate any control flow graph with a single
// entry and single exit block into structured control flow operations
// consisting of regions of do-while loops and operations conditionally
// dispatching to one out of multiple regions before continuing after the
// operation. This includes control flow graphs containing irreducible
// control flow.
//
// The implementation here additionally supports the transformation on
// regions with multiple exit blocks. This is implemented by first
// transforming all occurrences of return-like operations to branch to a
// single exit block containing an instance of that return-like operation.
// If there are multiple kinds of return-like operations, multiple exit
// blocks are created. In that case the transformation leaves behind a
// conditional control flow graph operating dispatching to the given regions
// terminating with different kinds of return-like operations each.
//
// If the function only contains a single kind of return-like operations,
// it is guaranteed to lift all control flow graph ops to structured control
// flow and that no more control flow graph ops remain after the operation.
//
// The algorithm to lift CFGs consists of two transformations applied after each
// other on any single-entry, single-exit region:
// 1) Lifting cycles to structured control flow loops
// 2) Lifting conditional branches to structured control flow branches
// These are then applied recursively on any new single-entry single-exit
// regions created by the transformation until no more CFG operations remain.
//
// The first part of cycle lifting is to detect any cycles in the CFG.
// This is done using an algorithm for iterating over SCCs. Every SCC
// representing a cycle is then transformed into a structured loop with a single
// entry block and a single latch containing the only back edge to the entry
// block and the only edge to an exit block outside the loop. Rerouting control
// flow to create single entry and exit blocks is achieved via a multiplexer
// construct best visualized as follows:
//                         +-----+ +-----+   +-----+
//                         | bb0 | | bb1 |...| bbN |
//                         +--+--+ +--+--+   +-+---+
//                            |       |        |
//                            |       v        |
//                            |  +------+      |
//                            | ++      ++<----+
//                            | | Region |
//                            +>|        |<----+
//                              ++      ++     |
//                               +------+------+
//
// transforms to:
//                         +-----+ +-----+   +-----+
//                         | bb0 | | bb1 |...| bbN |
//                         +-----+ +--|--+   ++----+
//                              |     v       |
//                              +->+-----+<---+
//                                 | bbM |<-------+
//                                 +---+-+        |
//                             +---+   | +----+   |
//                             |       v      |   |
//                             |   +------+   |   |
//                             |  ++      ++<-+   |
//                             +->| Region |      |
//                                ++      ++      |
//                                 +------+-------+
//
// bbM in the above is the multiplexer block and any block previously branching
// to an entry block of the region are redirected to it. This includes any
// branches from within the region. Using a block argument, bbM then dispatches
// to the correct entry block of the region dependent on the predecessor.
//
// A similar transformation is done to create the latch block with the single
// back edge and loop exit edge.
//
// The above form has the advantage that bbM now acts as the loop header
// of the loop body and after the transformation on the latch, results in a
// structured loop that can then be lifted to structured control flow. The
// conditional branches created in bbM are later lifted to conditional
// branches.
//
// Lifting conditional branches is done by analyzing the first conditional
// branch encountered in the entry region. The algorithm then identifies
// all blocks that are dominated by a specific control flow edge and
// the region where control flow continues:
//                                 +-----+
//                           +-----+ bb0 +----+
//                           v     +-----+    v
//                Region 1 +-+-+    ...     +-+-+ Region n
//                         +---+            +---+
//                          ...              ...
//                           |                |
//                           |      +---+     |
//                           +---->++   ++<---+
//                                 |     |
//                                 ++   ++ Region T
//                                  +---+
// Every region following bb0 consists of 0 or more blocks that eventually
// branch to Region T. If there are multiple entry blocks into Region T, a
// single entry block is created using a multiplexer block as shown above.
// Region 1 to Region n are then lifted together with the conditional control
// flow operation terminating bb0 into a structured conditional operation
// followed by the operations of the entry block of Region T.
//===----------------------------------------------------------------------===//

#include "mlir/Transforms/CFGToSCF.h"

#include "mlir/IR/RegionGraphTraits.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "llvm/ADT/DepthFirstIterator.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SCCIterator.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"

using namespace mlir;

/// Returns the mutable operand range used to transfer operands from `block` to
/// its successor with the given index. The return range being mutable allows
/// modifying the operands being transferred.
static MutableOperandRange
getMutableSuccessorOperands(Block *block, unsigned successorIndex) {
  auto branchOpInterface = cast<BranchOpInterface>(block->getTerminator());
  SuccessorOperands succOps =
      branchOpInterface.getSuccessorOperands(successorIndex);
  return succOps.getMutableForwardedOperands();
}

/// Appends all the block arguments from `other` to the block arguments of
/// `block`, copying their types and locations.
static void addBlockArgumentsFromOther(Block *block, Block *other) {
  block->addArguments(
      other->getArgumentTypes(),
      llvm::map_to_vector(other->getArguments(),
                          [](BlockArgument arg) { return arg.getLoc(); }));
}

namespace {

/// Class representing an edge in the CFG. Consists of a from-block, a successor
/// and corresponding successor operands passed to the block arguments of the
/// successor.
class Edge {
  Block *fromBlock;
  unsigned successorIndex;

public:
  /// Constructs a new edge from `fromBlock` to the successor corresponding to
  /// `successorIndex`.
  Edge(Block *fromBlock, unsigned int successorIndex)
      : fromBlock(fromBlock), successorIndex(successorIndex) {}

  /// Returns the from-block.
  Block *getFromBlock() const { return fromBlock; }

  /// Returns the successor of the edge.
  Block *getSuccessor() const {
    return fromBlock->getSuccessor(successorIndex);
  }

  /// Sets the successor of the edge, adjusting the terminator in the
  /// from-block.
  void setSuccessor(Block *block) const {
    fromBlock->getTerminator()->setSuccessor(block, successorIndex);
  }

  /// Returns the arguments of this edge that are passed to the block arguments
  /// of the successor.
  MutableOperandRange getSuccessorOperands() const {
    return getMutableSuccessorOperands(fromBlock, successorIndex);
  }
};

/// Returns a range of all edges from `block` to each of its successors.
auto successorEdges(Block *block) {
  return llvm::map_range(llvm::seq(block->getNumSuccessors()),
                         [=](unsigned index) { return Edge(block, index); });
}

/// Structure containing all special edges of a cycle. A cycle is a
/// generalization of a loop that may have multiple entry edges. See also
/// https://llvm.org/docs/CycleTerminology.html.
struct CyclesEdges {
  /// All edges from a block outside the cycle to a block inside the cycle.
  /// The targets of these edges are entry blocks.
  SmallVector<Edge> entryEdges;
  /// All edges from a block inside the cycle to a block outside the cycle.
  SmallVector<Edge> exitEdges;
  /// All edges from a block inside the cycle to an entry block.
  SmallVector<Edge> backEdges;
};

/// Class used to orchestrate creation of so-called edge multiplexers.
/// This constructs creates a new basic blocks and redirects all edges given to
/// be routed through that basic block before branching to their original
/// target. The purpose of this transformation is to create single-entry
/// single-exit regions by rerouting multiple entry edges or multiple exit edges
/// through an edge multiplexer.
class EdgeMultiplexer {
  Block *multiplexerBlock;
  function_ref<Value(unsigned)> getSwitchValue;
  function_ref<Value(Type)> getUndefValue;

  struct BlockEntry {
    unsigned offset;
    unsigned size;
  };

  llvm::SmallMapVector<Block *, BlockEntry, 4> entries;
  Value discriminator;

  EdgeMultiplexer(Block *multiplexerBlock,
                  function_ref<Value(unsigned)> getSwitchValue,
                  function_ref<Value(Type)> getUndefValue,
                  llvm::SmallMapVector<Block *, BlockEntry, 4> &&entries,
                  Value dispatchFlag)
      : multiplexerBlock(multiplexerBlock), getSwitchValue(getSwitchValue),
        getUndefValue(getUndefValue), entries(std::move(entries)),
        discriminator(dispatchFlag) {}

public:
  /// Creates a new edge multiplexer capable of redirecting all edges to one of
  /// the `entryBlocks`. This creates the multiplexer basic block with
  /// appropriate block arguments after the first entry block. `extraArgs`
  /// allows adding additional block arguments after the required ones to the
  /// multiplexer block, allowing additional successor operands to be passed
  /// from every edge.
  ///
  /// NOTE: This does not yet redirect edges to branch to the
  /// multiplexer block nor code dispatching from the multiplexer code
  /// to the original successors.
  /// See `redirectEdge` and `createSwitch`.
  static EdgeMultiplexer
  create(Location loc, ArrayRef<Block *> entryBlocks,
         llvm::function_ref<Value(unsigned)> getSwitchValue,
         llvm::function_ref<Value(Type)> getUndefValue,
         TypeRange extraArgs = {}) {
    assert(!entryBlocks.empty() && "Require at least one entry block");

    auto *multiplexerBlock = new Block;
    multiplexerBlock->insertAfter(entryBlocks.front());

    // To implement the multiplexer block, we have to add the block arguments of
    // every distinct successor block to the multiplexer block. When redirecting
    // edges, block arguments designated for blocks that aren't branched to will
    // be assigned the `getUndefValue`. The amount of block arguments and their
    // offset is saved in the map for `redirectEdges` to transform the edges.
    llvm::SmallMapVector<Block *, BlockEntry, 4> entries;
    for (Block *entryBlock : entryBlocks) {
      auto [iter, inserted] =
          entries.insert({entryBlock,
                          {/*offset=*/multiplexerBlock->getNumArguments(),
                           /*size=*/entryBlock->getNumArguments()}});
      if (inserted)
        addBlockArgumentsFromOther(multiplexerBlock, entryBlock);
    }

    // If we have more than one successor, we have to additionally add a
    // discriminator value, denoting which successor to jump to.
    // When redirecting edges, an appropriate value will be passed using
    // `getSwitchValue`.
    Value discriminator;
    if (entries.size() > 1)
      discriminator =
          multiplexerBlock->addArgument(getSwitchValue(0).getType(), loc);

    multiplexerBlock->addArguments(
        extraArgs, SmallVector<Location>(extraArgs.size(), loc));

    return EdgeMultiplexer(multiplexerBlock, getSwitchValue, getUndefValue,
                           std::move(entries), discriminator);
  }

  /// Returns the created multiplexer block.
  Block *getMultiplexerBlock() const { return multiplexerBlock; }

  /// Redirects `edge` to branch to the multiplexer block before continuing to
  /// its original target. The edges successor must have originally been part
  /// of the entry blocks array passed to the `create` function. `extraArgs`
  /// must be used to pass along any additional values corresponding to
  /// `extraArgs` in `create`.
  void redirectEdge(Edge edge, ValueRange extraArgs = {}) const {
    const auto *result = entries.find(edge.getSuccessor());
    assert(result != entries.end() &&
           "Edge was not originally passed to `create` method.");

    const BlockEntry &entry = result->second;

    SmallVector<Value> newSuccOperands(multiplexerBlock->getNumArguments());
    auto *iter = newSuccOperands.begin();

    // First, create undefs for all block arguments that aren't for our
    // successor block.
    ValueRange destArgs = multiplexerBlock->getArguments();
    iter = llvm::transform(
        destArgs.take_front(entry.offset), iter,
        [&](Value arg) { return getUndefValue(arg.getType()); });
    destArgs = destArgs.drop_front(entry.offset);

    // Now copy the original successor arguments.
    MutableOperandRange arguments = edge.getSuccessorOperands();
    iter = llvm::copy(arguments, iter);
    destArgs = destArgs.drop_front(arguments.size());

    unsigned remainingArgsSize = discriminator ? 1 : 0;
    remainingArgsSize += extraArgs.size();
    // Followed by again creating undefs for the block arguments of all other
    // blocks we aren't jumping to.
    iter = llvm::transform(
        destArgs.drop_back(remainingArgsSize), iter,
        [&](Value arg) { return getUndefValue(arg.getType()); });

    // Last, pass along the discriminator and any extra arguments.
    if (discriminator) {
      *iter = getSwitchValue(result - entries.begin());
      iter = std::next(iter);
    }

    llvm::copy(extraArgs, iter);

    edge.setSuccessor(multiplexerBlock);
    arguments.assign(newSuccOperands);
  }

  /// Creates a switch op using `builder` which dispatches to the original
  /// successors of the edges passed to `create` minus the ones in `excluded`.
  /// The builder's insertion point has to be in a block dominated by the
  /// multiplexer block.
  void createSwitch(
      Location loc, OpBuilder &builder, CFGToSCFInterface &interface,
      const SmallPtrSetImpl<Block *> &excluded = SmallPtrSet<Block *, 1>{}) {
    // We create the switch by creating a case for all entries and then
    // splitting of the last entry as a default case.

    SmallVector<ValueRange> caseArguments;
    SmallVector<unsigned> caseValues;
    SmallVector<Block *> caseDestinations;

    for (auto &&[index, pair] : llvm::enumerate(entries)) {
      auto &&[succ, entry] = pair;
      if (excluded.contains(succ))
        continue;

      caseValues.push_back(index);
      caseArguments.push_back(
          multiplexerBlock->getArguments().slice(entry.offset, entry.size));
      caseDestinations.push_back(succ);
    }

    // If we don't have a discriminator due to only having one entry we have to
    // create a dummy flag for the switch.
    Value realDiscriminator = discriminator && caseArguments.size() > 1
                                  ? discriminator
                                  : getSwitchValue(0);

    caseValues.pop_back();
    Block *defaultDest = caseDestinations.pop_back_val();
    ValueRange defaultArgs = caseArguments.pop_back_val();

    interface.createCFGSwitchOperation(loc, builder, realDiscriminator,
                                       caseValues, caseDestinations,
                                       caseArguments, defaultDest, defaultArgs);
  }
};

/// Alternative implementation of DenseMapInfo<Operation*> using the operation
/// equivalence infrastructure to check whether two 'return-like' operations are
/// equivalent in the context of this transformation. This means that both
/// operations are of the same kind, have the same amount of operands and types
/// and the same attributes and properties. The operands themselves don't have
/// to be equivalent.
struct ReturnLikeOperationEquivalence : public llvm::DenseMapInfo<Operation *> {
  static unsigned getHashValue(const Operation *opC) {
    return OperationEquivalence::computeHash(
        const_cast<Operation *>(opC),
        /*hashOperands=*/OperationEquivalence::ignoreHashValue,
        /*hashResults=*/OperationEquivalence::ignoreHashValue,
        OperationEquivalence::IgnoreLocations);
  }

  static bool isEqual(const Operation *lhs, const Operation *rhs) {
    if (lhs == rhs)
      return true;
    if (lhs == getTombstoneKey() || lhs == getEmptyKey() ||
        rhs == getTombstoneKey() || rhs == getEmptyKey())
      return false;
    return OperationEquivalence::isEquivalentTo(
        const_cast<Operation *>(lhs), const_cast<Operation *>(rhs),
        OperationEquivalence::ignoreValueEquivalence, nullptr,
        OperationEquivalence::IgnoreLocations);
  }
};

/// Utility-class for transforming a region to only have one single block for
/// every return-like operation.
class ReturnLikeExitTransformer {
  // Mapping of return-like operation to block. All return-like operations
  // of the same kind with the same attributes, properties and types are seen as
  // equivalent. First occurrence seen is kept in the map.
  llvm::SmallDenseMap<Operation *, Block *, 4, ReturnLikeOperationEquivalence>
      mapping;
  Region &topLevelRegion;
  CFGToSCFInterface &interface;

public:
  ReturnLikeExitTransformer(Region &topLevelRegion,
                            CFGToSCFInterface &interface)
      : topLevelRegion(topLevelRegion), interface(interface) {}

  /// If more than one kind of `returnLikeOperation` has been seen, transforms
  /// `returnLikeOperation` to a branch to the only block in region with an
  /// instance of `returnLikeOperation`s kind.
  void transform(Operation *returnLikeOperation,
                 llvm::function_ref<Value(unsigned)> getSwitchValue) {
    // First insertion does not cause any transformation.
    auto [iter, inserted] = mapping.insert({returnLikeOperation, nullptr});
    if (inserted)
      return;

    // If this is the second occurrence, create the exit-block for the
    // operation-kind and redirect the first seen occurrence as well.
    Block *exitBlock = iter->second;
    if (!exitBlock) {
      exitBlock = new Block;
      iter->second = exitBlock;
      topLevelRegion.push_back(exitBlock);
      Operation *originalOperation = iter->first;
      exitBlock->addArguments(
          originalOperation->getOperandTypes(),
          SmallVector<Location>(originalOperation->getNumOperands(),
                                originalOperation->getLoc()));

      auto builder =
          OpBuilder::atBlockTerminator(originalOperation->getBlock());
      interface.createSingleDestinationBranch(
          originalOperation->getLoc(), builder, getSwitchValue(0), exitBlock,
          originalOperation->getOperands());
      originalOperation->moveBefore(exitBlock, exitBlock->end());
      originalOperation->setOperands(exitBlock->getArguments());
    }

    auto builder =
        OpBuilder::atBlockTerminator(returnLikeOperation->getBlock());
    interface.createSingleDestinationBranch(
        returnLikeOperation->getLoc(), builder, getSwitchValue(0), exitBlock,
        returnLikeOperation->getOperands());
    returnLikeOperation->erase();
  }

  /// Returns true if any IR transformations have occurred so far.
  bool changed() const {
    // IR transformations have occurred if any exit blocks were created.
    return llvm::any_of(llvm::make_second_range(mapping),
                        llvm::identity<Block *>{});
  }
};

} // namespace

/// Calculates all special edges of the given cycle.
static CyclesEdges
calculateCycleEdges(const llvm::SmallSetVector<Block *, 4> &cycles) {
  CyclesEdges result;
  SmallPtrSet<Block *, 8> entryBlock;

  // First identify all exit and entry edges by checking whether any successors
  // or predecessors are from outside the cycles.
  for (Block *block : cycles) {
    for (auto pred = block->pred_begin(); pred != block->pred_end(); pred++) {
      if (cycles.contains(*pred))
        continue;

      result.entryEdges.emplace_back(*pred, pred.getSuccessorIndex());
      entryBlock.insert(block);
    }

    for (auto &&[succIndex, succ] : llvm::enumerate(block->getSuccessors())) {
      if (cycles.contains(succ))
        continue;

      result.exitEdges.emplace_back(block, succIndex);
    }
  }

  // With the entry blocks identified, find all the back edges.
  for (Block *block : cycles)
    for (auto &&[succIndex, succ] : llvm::enumerate(block->getSuccessors())) {
      if (!entryBlock.contains(succ))
        continue;

      result.backEdges.emplace_back(block, succIndex);
    }

  return result;
}

/// Creates a single entry block out of multiple entry edges using a edge
/// multiplexer and returns it.
static EdgeMultiplexer
createSingleEntryBlock(Location loc, ArrayRef<Edge> entryEdges,
                       llvm::function_ref<Value(unsigned)> getSwitchValue,
                       llvm::function_ref<Value(Type)> getUndefValue,
                       CFGToSCFInterface &interface) {
  auto result = EdgeMultiplexer::create(
      loc, llvm::map_to_vector(entryEdges, std::mem_fn(&Edge::getSuccessor)),
      getSwitchValue, getUndefValue);

  auto builder = OpBuilder::atBlockBegin(result.getMultiplexerBlock());
  result.createSwitch(loc, builder, interface);

  for (Edge edge : entryEdges)
    result.redirectEdge(edge);

  return result;
}

/// Special loop properties of a structured loop.
/// A structured loop is a loop satisfying all of the following:
/// * Has at most one entry, exit and back edge.
/// * The back edge originates from the same block as the exit edge.
struct StructuredLoopProperties {
  /// Block containing both the single exit edge and the single back edge.
  Block *latch;
  /// Loop condition of type equal to a value returned by `getSwitchValue`.
  Value condition;
  /// Exit block which is the only successor of the loop.
  Block *exitBlock;
};

/// Transforms a loop into a structured loop with only a single back edge and
/// exiting edge, originating from the same block.
static StructuredLoopProperties createSingleExitingLatch(
    Location loc, ArrayRef<Edge> backEdges, ArrayRef<Edge> exitEdges,
    llvm::function_ref<Value(unsigned)> getSwitchValue,
    llvm::function_ref<Value(Type)> getUndefValue, CFGToSCFInterface &interface,
    ReturnLikeExitTransformer &exitTransformer) {
  assert(llvm::all_equal(
             llvm::map_range(backEdges, std::mem_fn(&Edge::getSuccessor))) &&
         "All repetition arcs must lead to the single loop header");

  // First create the multiplexer block, which will be our latch, for all back
  // edges and exit edges. We pass an additional argument to the multiplexer
  // block which indicates whether the latch was reached from what was
  // originally a back edge or an exit block.
  // This is later used to branch using the new only back edge.
  SmallVector<Block *> successors;
  llvm::append_range(
      successors, llvm::map_range(backEdges, std::mem_fn(&Edge::getSuccessor)));
  llvm::append_range(
      successors, llvm::map_range(exitEdges, std::mem_fn(&Edge::getSuccessor)));
  auto multiplexer =
      EdgeMultiplexer::create(loc, successors, getSwitchValue, getUndefValue,
                              /*extraArgs=*/getSwitchValue(0).getType());

  auto *latchBlock = multiplexer.getMultiplexerBlock();

  // Create a separate exit block that comes right after the latch.
  auto *exitBlock = new Block;
  exitBlock->insertAfter(latchBlock);

  // Since this is a loop, all back edges point to the same loop header.
  Edge latchRepr = backEdges.front();
  Block *loopHeader = latchRepr.getSuccessor();

  // Create the new only back edge to the loop header. Branch to the
  // exit block otherwise.
  Value shouldRepeat = latchBlock->getArguments().back();
  {
    auto builder = OpBuilder::atBlockBegin(latchBlock);
    interface.createConditionalBranch(
        builder.getUnknownLoc(), builder, shouldRepeat, loopHeader,
        latchBlock->getArguments().take_front(loopHeader->getNumArguments()),
        /*falseDest=*/exitBlock,
        /*falseArgs=*/{});
  }

  {
    auto builder = OpBuilder::atBlockBegin(exitBlock);
    if (!exitEdges.empty()) {
      // Create the switch dispatching to what were originally the multiple exit
      // blocks. The loop header has to explicitly be excluded in the below
      // switch as we would otherwise be creating a new loop again. All back
      // edges leading to the loop header have already been handled in the
      // switch above. The remaining edges can only jump to blocks outside the
      // loop.

      SmallPtrSet<Block *, 1> excluded = {loopHeader};
      multiplexer.createSwitch(loc, builder, interface, excluded);
    } else {
      // A loop without an exit edges is a statically known infinite loop.
      // Since structured control flow ops are not terminator ops, the caller
      // has to create a fitting return-like unreachable terminator operation.
      Operation *terminator = interface.createUnreachableTerminator(
          loc, builder, *latchBlock->getParent());
      // Transform the just created transform operation in the case that an
      // occurrence of it existed in input IR.
      exitTransformer.transform(terminator, getSwitchValue);
    }
  }

  // Redirecting back edges with `shouldRepeat` as 1.
  for (Edge backEdge : backEdges)
    multiplexer.redirectEdge(backEdge, /*extraArgs=*/getSwitchValue(1));

  // Redirecting exits edges with `shouldRepeat` as 0.
  for (Edge exitEdge : exitEdges)
    multiplexer.redirectEdge(exitEdge, /*extraArgs=*/getSwitchValue(0));

  return {latchBlock, /*condition=*/shouldRepeat, exitBlock};
}

/// Transforms a structured loop into a loop in reduce form.
///
/// Reduce form is defined as a structured loop where:
/// (0) No values defined within the loop body are used outside the loop body.
/// (1) The block arguments and successor operands of the exit block are equal
///     to the block arguments of the loop header and the successor operands
///     of the back edge.
///
/// This is required for many structured control flow ops as they tend
/// to not have separate "loop result arguments" and "loop iteration arguments"
/// at the end of the block. Rather, the "loop iteration arguments" from the
/// last iteration are the result of the loop.
///
/// Note that the requirement of (0) is equal to LCSSA form in LLVM. However,
/// due to this being a structured loop instead of a general loop, we do not
/// require complicated dominance algorithms nor SSA updating making this
/// implementation easier than creating a generic LCSSA transformation pass.
static SmallVector<Value>
transformToReduceLoop(Block *loopHeader, Block *exitBlock,
                      const llvm::SmallSetVector<Block *, 4> &loopBlocks,
                      llvm::function_ref<Value(Type)> getUndefValue) {
  Block *latch = exitBlock->getSinglePredecessor();
  assert(latch &&
         "Exit block must have only latch as predecessor at this point");
  assert(exitBlock->getNumArguments() == 0 &&
         "Exit block mustn't have any block arguments at this point");

  std::array<MutableOperandRange, 2> latchSuccessorOperands = [&] {
    unsigned loopHeaderIndex = 0;
    unsigned exitBlockIndex = 1;
    if (latch->getSuccessor(loopHeaderIndex) != loopHeader)
      std::swap(loopHeaderIndex, exitBlockIndex);

    assert(latch->getSuccessor(loopHeaderIndex) == loopHeader);
    assert(latch->getSuccessor(exitBlockIndex) == exitBlock);

    return std::array{getMutableSuccessorOperands(latch, exitBlockIndex),
                      getMutableSuccessorOperands(latch, loopHeaderIndex)};
  }();

  auto &&[exitBlockSuccessorOperands, loopHeaderSuccessorOperands] =
      latchSuccessorOperands;

  // Add all iteration variables of the loop header to the exit block. Replace
  // any uses that are outside the loop with the newly created exit block.
  for (BlockArgument arg : loopHeader->getArguments()) {
    BlockArgument exitArg = exitBlock->addArgument(arg.getType(), arg.getLoc());
    exitBlockSuccessorOperands.append(
        loopHeaderSuccessorOperands[arg.getArgNumber()]);
    arg.replaceUsesWithIf(exitArg, [&](OpOperand &use) {
      return !loopBlocks.contains(use.getOwner()->getBlock());
    });
  }

  // Go over all values defined within the loop body. If any of them are used
  // outside the loop body, create a block argument on the exit block and loop
  // header and replace the outside uses with the exit block argument.
  // The loop header block argument is added to satisfy requirement (1) in the
  // reduce form condition.
  for (Block *loopBlock : loopBlocks) {

    auto checkValue = [&](Value value) {
      Value blockArgument;
      for (OpOperand &use : llvm::make_early_inc_range(value.getUses())) {
        if (loopBlocks.contains(use.getOwner()->getBlock()))
          continue;

        // Block argument is only created the first time it is required.
        if (!blockArgument) {
          blockArgument =
              exitBlock->addArgument(value.getType(), value.getLoc());
          loopHeader->addArgument(value.getType(), value.getLoc());

          for (MutableOperandRange &range : latchSuccessorOperands)
            range.append(value);
        }

        use.set(blockArgument);
      }
    };

    // Loop header block arguments were already handled above.
    if (loopBlock != loopHeader) {
      llvm::for_each(loopBlock->getArguments(), checkValue);
      continue;
    }

    for (Operation &op : *loopBlock)
      llvm::for_each(op.getResults(), checkValue);
  }

  // New block arguments may have been added to the loop header.
  // Adjust the entry edges to pass undef values to these.
  for (auto iter = loopHeader->pred_begin(); iter != loopHeader->pred_end();
       iter++) {
    // Latch successor arguments have already been handled.
    if (*iter == latch)
      continue;

    MutableOperandRange succOps =
        getMutableSuccessorOperands(*iter, iter.getSuccessorIndex());
    succOps.append(llvm::map_to_vector(
        loopHeader->getArguments().drop_front(succOps.size()),
        [&](BlockArgument arg) { return getUndefValue(arg.getType()); }));
  }

  return static_cast<OperandRange>(loopHeaderSuccessorOperands);
}

/// Transforms all outer-most cycles in the region with the region entry
/// `regionEntry` into structured loops. Returns the entry blocks of any newly
/// created regions potentially requiring further transformations.
static SmallVector<Block *> transformCyclesToSCFLoops(
    Block *regionEntry, llvm::function_ref<Value(unsigned)> getSwitchValue,
    llvm::function_ref<Value(Type)> getUndefValue, CFGToSCFInterface &interface,
    ReturnLikeExitTransformer &exitTransformer) {
  SmallVector<Block *> newSubRegions;
  auto scc = llvm::scc_begin(regionEntry);
  while (!scc.isAtEnd()) {
    if (!scc.hasCycle()) {
      ++scc;
      continue;
    }

    // Save the set and increment the SCC iterator early to avoid our
    // modifications breaking the SCC iterator.
    llvm::SmallSetVector<Block *, 4> cycleBlockSet(scc->begin(), scc->end());
    ++scc;

    CyclesEdges arcs = calculateCycleEdges(cycleBlockSet);
    Block *loopHeader = arcs.entryEdges.front().getSuccessor();
    // First turn the cycle into a loop by creating a single entry block if
    // needed.
    if (arcs.entryEdges.size() > 1) {
      EdgeMultiplexer entryArcMapping = createSingleEntryBlock(
          loopHeader->getTerminator()->getLoc(), arcs.entryEdges,
          getSwitchValue, getUndefValue, interface);

      for (Edge edge : arcs.backEdges)
        entryArcMapping.redirectEdge(edge);

      loopHeader = entryArcMapping.getMultiplexerBlock();
    }

    // Then turn it into a structured loop by creating a single latch.
    StructuredLoopProperties loopProperties = createSingleExitingLatch(
        arcs.backEdges.front().getFromBlock()->getTerminator()->getLoc(),
        arcs.backEdges, arcs.exitEdges, getSwitchValue, getUndefValue,
        interface, exitTransformer);

    Block *latchBlock = loopProperties.latch;
    Block *exitBlock = loopProperties.exitBlock;
    cycleBlockSet.insert(latchBlock);
    cycleBlockSet.insert(loopHeader);

    // Finally, turn it into reduce form.
    ValueRange iterationValues = transformToReduceLoop(
        loopHeader, exitBlock, cycleBlockSet, getUndefValue);

    // Create a block acting as replacement for the loop header and insert
    // the structured loop into it.
    auto *newLoopHeader = new Block;
    newLoopHeader->insertBefore(loopHeader);
    addBlockArgumentsFromOther(newLoopHeader, loopHeader);

    Region::BlockListType &blocks = regionEntry->getParent()->getBlocks();
    Region loopBody;
    // Make sure the loop header is the entry block.
    loopBody.push_back(blocks.remove(loopHeader));
    for (Block *block : cycleBlockSet)
      if (block != latchBlock && block != loopHeader)
        loopBody.push_back(blocks.remove(block));
    // And the latch is the last block.
    loopBody.push_back(blocks.remove(latchBlock));

    Operation *oldTerminator = latchBlock->getTerminator();
    oldTerminator->remove();

    auto builder = OpBuilder::atBlockBegin(newLoopHeader);
    Operation *scfOp = interface.createDoWhileLoopOp(
        builder, oldTerminator, newLoopHeader->getArguments(),
        loopProperties.condition, iterationValues, std::move(loopBody));
    oldTerminator->erase();

    newSubRegions.push_back(loopHeader);

    for (auto &&[oldValue, newValue] :
         llvm::zip(exitBlock->getArguments(), scfOp->getResults()))
      oldValue.replaceAllUsesWith(newValue);

    loopHeader->replaceAllUsesWith(newLoopHeader);
    // Merge the exit block right after the loop operation.
    newLoopHeader->getOperations().splice(newLoopHeader->end(),
                                          exitBlock->getOperations());
    exitBlock->erase();
  }
  return newSubRegions;
}

/// Makes sure the branch region has only a single exit. This is required by the
/// recursive part of the algorithm as it expects the CFG to be single-entry
/// and single-exit. This is done by simply creating an empty block if there
/// is more than one block with an edge to the continuation block. All blocks
/// with edges to the continuation are then redirected to the block. A region
/// terminator is later placed into the block.
static void createSingleExitBranchRegion(
    ArrayRef<Block *> branchRegion, Block *continuation,
    SmallVectorImpl<std::pair<Block *, SmallVector<Value>>> &createdEmptyBlocks,
    Region &conditionalRegion) {
  Block *singleExitBlock = nullptr;
  std::optional<Edge> previousEdgeToContinuation;
  Region::BlockListType &parentBlockList =
      branchRegion.front()->getParent()->getBlocks();
  for (Block *block : branchRegion) {
    for (Edge edge : successorEdges(block)) {
      if (edge.getSuccessor() != continuation)
        continue;

      if (!previousEdgeToContinuation) {
        previousEdgeToContinuation = edge;
        continue;
      }

      // If this is not the first edge to the continuation we create the
      // single exit block and redirect the edges.
      if (!singleExitBlock) {
        singleExitBlock = new Block;
        addBlockArgumentsFromOther(singleExitBlock, continuation);
        previousEdgeToContinuation->setSuccessor(singleExitBlock);
        createdEmptyBlocks.emplace_back(singleExitBlock,
                                        singleExitBlock->getArguments());
      }

      edge.setSuccessor(singleExitBlock);
    }

    conditionalRegion.push_back(parentBlockList.remove(block));
  }

  if (singleExitBlock)
    conditionalRegion.push_back(singleExitBlock);
}

/// Transforms the first occurrence of conditional control flow in `regionEntry`
/// into conditionally executed regions. Returns the entry block of the created
/// regions and the region after the conditional control flow.
static SmallVector<Block *> transformToCFGBranches(
    Block *regionEntry, llvm::function_ref<Value(unsigned)> getSwitchValue,
    llvm::function_ref<Value(Type)> getUndefValue, CFGToSCFInterface &interface,
    DominanceInfo &dominanceInfo) {
  switch (regionEntry->getNumSuccessors()) {
  case 0:
    // Trivial region.
    return {};
  case 1: {
    // Single successor we can just splice together.
    Block *successor = regionEntry->getSuccessor(0);
    for (auto &&[oldValue, newValue] :
         llvm::zip(successor->getArguments(),
                   getMutableSuccessorOperands(regionEntry, 0)))
      oldValue.replaceAllUsesWith(newValue);
    regionEntry->getTerminator()->erase();

    regionEntry->getOperations().splice(regionEntry->end(),
                                        successor->getOperations());
    successor->erase();
    return {regionEntry};
  }
  default:
    break;
  }

  SmallVector<Block *> newSubRegions;

  // Split the CFG into "#numSuccessor + 1" regions.
  // For every edge to a successor, the blocks it solely dominates are
  // determined and become the region following that edge.
  // The last region is the continuation that follow the branch regions.
  SmallPtrSet<Block *, 8> notContinuation;
  notContinuation.insert(regionEntry);
  SmallVector<SmallVector<Block *>> successorBranchRegions(
      regionEntry->getNumSuccessors());
  for (auto &&[blockList, succ] :
       llvm::zip(successorBranchRegions, regionEntry->getSuccessors())) {
    // If the region entry is not the only predecessor, then the edge does not
    // dominate the block it leads to.
    if (succ->getSinglePredecessor() != regionEntry)
      continue;

    // Otherwise get all blocks it dominates in DFS/pre-order.
    DominanceInfoNode *node = dominanceInfo.getNode(succ);
    for (DominanceInfoNode *curr : llvm::depth_first(node)) {
      blockList.push_back(curr->getBlock());
      notContinuation.insert(curr->getBlock());
    }
  }

  // Determine all edges into the continuation. These are simply all edges from
  // a block in the branch region to a block in the continuation.
  SmallVector<Edge> continuationEdges;
  bool allSuccessorsHaveContinuationEdge = true;
  bool noSuccessorHasContinuationEdge = true;
  for (auto &&[entryEdge, branchRegion] :
       llvm::zip(successorEdges(regionEntry), successorBranchRegions)) {

    // If branch region is empty then the branch target itself is part of the
    // continuation.
    if (branchRegion.empty()) {
      continuationEdges.push_back(entryEdge);
      noSuccessorHasContinuationEdge = false;
      continue;
    }

    bool hadEdge = false;
    for (Block *block : branchRegion) {
      for (Edge edge : successorEdges(block)) {
        if (notContinuation.contains(edge.getSuccessor()))
          continue;

        continuationEdges.push_back(edge);
        hadEdge = true;
        noSuccessorHasContinuationEdge = false;
      }
    }

    if (!hadEdge) {
      allSuccessorsHaveContinuationEdge = false;
      continuationEdges.push_back(entryEdge);
    }
  }

  // There are three cases to differentiate here:
  // 1) All branch regions have continuation edges. This is the usual case.
  // 2) No branch region has a continuation edge. This will occur if we are in
  // the top-level region and have multiple different return-like operations. In
  // that case the control flow operation must stay as we are unable to create a
  // single exit-block. We can nevertheless process all its successors as they
  // are all single-entry regions.
  // 3) Some branch regions have an edge to one or more continuation.
  // This occurs if some branches end in the same return-like operation and some
  // don't. In that case we cannot immediately create a structured control
  // flow op either, but we can create a single entry block for all branch
  // regions and reprocess. The single entry block will then act as a single
  // continuation allowing us to create a structured control flow op. The single
  // entry block itself also has exactly one outgoing-edge to each region
  // terminating in a different return-like operation causing case 2 to apply.
  if (noSuccessorHasContinuationEdge)
    return regionEntry->getSuccessors();

  Block *continuation = llvm::find_singleton<Block>(
      continuationEdges, [](Edge edge, bool) { return edge.getSuccessor(); },
      /*AllowRepeats=*/true);

  // In case 3) or if not all continuation edges have the same entry block,
  // create a single entry block as continuation for all branch regions.
  if (!continuation || !allSuccessorsHaveContinuationEdge) {
    EdgeMultiplexer multiplexer = createSingleEntryBlock(
        continuationEdges.front().getFromBlock()->getTerminator()->getLoc(),
        continuationEdges, getSwitchValue, getUndefValue, interface);
    continuation = multiplexer.getMultiplexerBlock();
  }

  // Trigger reprocess of case 3) after creating single entry block.
  if (!allSuccessorsHaveContinuationEdge)
    return {regionEntry};

  // Empty blocks with the values they return to the parent op.
  SmallVector<std::pair<Block *, SmallVector<Value>>> createdEmptyBlocks;

  // Create the branch regions.
  std::vector<Region> conditionalRegions(successorBranchRegions.size());
  for (auto &&[branchRegion, entryEdge, conditionalRegion] :
       llvm::zip(successorBranchRegions, successorEdges(regionEntry),
                 conditionalRegions)) {
    if (branchRegion.empty()) {
      // If no block is part of the branch region, we create a dummy block to
      // place the region terminator into.
      createdEmptyBlocks.emplace_back(
          new Block, llvm::to_vector(entryEdge.getSuccessorOperands()));
      conditionalRegion.push_back(createdEmptyBlocks.back().first);
      continue;
    }

    createSingleExitBranchRegion(branchRegion, continuation, createdEmptyBlocks,
                                 conditionalRegion);

    // The entries of the branch regions may only have redundant block arguments
    // since the edge to the branch region is always dominating.
    Block *regionEntryBlock = &conditionalRegion.front();
    for (auto &&[oldValue, newValue] :
         llvm::zip(regionEntryBlock->getArguments(),
                   entryEdge.getSuccessorOperands()))
      oldValue.replaceAllUsesWith(newValue);

    regionEntryBlock->eraseArguments(0, regionEntryBlock->getNumArguments());
    newSubRegions.push_back(regionEntryBlock);
  }

  Operation *condOp;
  {
    auto opBuilder = OpBuilder::atBlockTerminator(regionEntry);
    condOp = interface.createBranchRegionOp(
        opBuilder, regionEntry->getTerminator(),
        continuation->getArgumentTypes(), conditionalRegions);
    regionEntry->getTerminator()->erase();
  }

  for (auto &&[block, valueRange] : createdEmptyBlocks) {
    auto builder = OpBuilder::atBlockEnd(block);
    interface.createBranchRegionTerminatorOp(condOp->getLoc(), builder, condOp,
                                             valueRange);
  }

  // Any leftover users of the continuation must be from unconditional branches
  // in a branch region. There can only be at most one per branch region as
  // all branch regions have been made single-entry single-exit above.
  // Replace them with the region terminator.
  for (Operation *user : llvm::make_early_inc_range(continuation->getUsers())) {
    assert(user->getNumSuccessors() == 1);
    auto builder = OpBuilder::atBlockTerminator(user->getBlock());
    interface.createBranchRegionTerminatorOp(
        user->getLoc(), builder, condOp,
        static_cast<OperandRange>(
            getMutableSuccessorOperands(user->getBlock(), 0)));
    user->erase();
  }

  for (auto &&[oldValue, newValue] :
       llvm::zip(continuation->getArguments(), condOp->getResults()))
    oldValue.replaceAllUsesWith(newValue);

  // Splice together the continuations operations with the region entry.
  regionEntry->getOperations().splice(regionEntry->end(),
                                      continuation->getOperations());

  continuation->erase();

  // After splicing the continuation, the region has to be reprocessed as it has
  // new successors.
  newSubRegions.push_back(regionEntry);

  return newSubRegions;
}

/// Transforms the region to only have a single block for every kind of
/// return-like operation that all previous occurrences of the return-like op
/// branch to. If the region only contains a single kind of return-like
/// operation, it creates a single-entry and single-exit region.
static ReturnLikeExitTransformer createSingleExitBlocksForReturnLike(
    Region &region, llvm::function_ref<Value(unsigned)> getSwitchValue,
    CFGToSCFInterface &interface) {
  ReturnLikeExitTransformer exitBlockMapping(region, interface);

  for (Operation &operation : llvm::make_early_inc_range(region.getOps())) {
    if (!operation.hasTrait<OpTrait::ReturnLike>())
      continue;
    exitBlockMapping.transform(&operation, getSwitchValue);
  }

  return exitBlockMapping;
}

/// Checks all preconditions of the transformation prior to any transformations.
/// Returns failure if any precondition is violated.
static LogicalResult checkTransformationPreconditions(Region &region) {
  WalkResult result = region.walk([](Operation *operation) {
    if (operation->getNumSuccessors() == 0)
      return WalkResult::advance();

    // This transformation requires all ops with successors to implement the
    // branch op interface. It is impossible to adjust their block arguments
    // otherwise.
    auto branchOpInterface = dyn_cast<BranchOpInterface>(operation);
    if (!branchOpInterface) {
      operation->emitOpError("transformation does not support terminators with "
                             "successors not implementing BranchOpInterface");
      return WalkResult::interrupt();
    }
    // Branch operations must have no side effects. Replacing them would not be
    // valid otherwise.
    if (!isMemoryEffectFree(branchOpInterface)) {
      branchOpInterface->emitOpError(
          "transformation does not support terminators with side effects");
      return WalkResult::interrupt();
    }

    for (unsigned index : llvm::seq(operation->getNumSuccessors())) {
      SuccessorOperands succOps = branchOpInterface.getSuccessorOperands(index);

      // We cannot support operations with operation-produced successor operands
      // as it is currently not possible to pass them to any block arguments
      // other than the first. This breaks creating multiplexer blocks and would
      // likely need special handling elsewhere too.
      if (succOps.getProducedOperandCount() == 0)
        continue;

      branchOpInterface->emitOpError("transformation does not support "
                                     "operations with operation-produced "
                                     "successor operands");
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });
  return failure(result.wasInterrupted());
}

FailureOr<bool> mlir::transformCFGToSCF(Region &region,
                                        CFGToSCFInterface &interface,
                                        DominanceInfo &dominanceInfo) {
  if (failed(checkTransformationPreconditions(region)))
    return failure();

  auto constantBuilder = OpBuilder::atBlockBegin(&region.front());

  DenseMap<Type, Value> typedUndefCache;
  auto getUndefValue = [&](Type type) {
    auto [iter, inserted] = typedUndefCache.insert({type, nullptr});
    if (!inserted)
      return iter->second;

    iter->second =
        interface.getUndefValue(region.getLoc(), constantBuilder, type);
    return iter->second;
  };

  // The transformation only creates all values in the range of 0 to
  // max(#numSuccessors). Therefore using a vector instead of a map.
  SmallVector<Value> switchValueCache;
  auto getSwitchValue = [&](unsigned value) {
    if (value < switchValueCache.size())
      if (switchValueCache[value])
        return switchValueCache[value];

    switchValueCache.resize(
        std::max<size_t>(switchValueCache.size(), value + 1));

    switchValueCache[value] =
        interface.getCFGSwitchValue(region.getLoc(), constantBuilder, value);
    return switchValueCache[value];
  };

  ReturnLikeExitTransformer exitTransformer =
      createSingleExitBlocksForReturnLike(region, getSwitchValue, interface);
  bool changed = exitTransformer.changed();

  SmallVector<Block *> workList = {&region.front()};
  while (!workList.empty()) {
    Block *current = workList.pop_back_val();

    size_t sizePrior = workList.size();
    llvm::append_range(workList, transformCyclesToSCFLoops(
                                     current, getSwitchValue, getUndefValue,
                                     interface, exitTransformer));
    llvm::append_range(
        workList, transformToCFGBranches(current, getSwitchValue, getUndefValue,
                                         interface, dominanceInfo));
    if (sizePrior != workList.size())
      changed = true;
  }

  return changed;
}
