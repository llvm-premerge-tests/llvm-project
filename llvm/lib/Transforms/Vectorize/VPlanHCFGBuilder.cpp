//===-- VPlanHCFGBuilder.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements the construction of a VPlan-based Hierarchical CFG
/// (H-CFG) for an incoming IR. This construction comprises the following
/// components and steps:
//
/// 1. PlainCFGBuilder class: builds a plain VPBasicBlock-based CFG that
/// faithfully represents the CFG in the incoming IR. A VPRegionBlock (Top
/// Region) is created to enclose and serve as parent of all the VPBasicBlocks
/// in the plain CFG.
/// NOTE: At this point, there is a direct correspondence between all the
/// VPBasicBlocks created for the initial plain CFG and the incoming
/// BasicBlocks. However, this might change in the future.
///
//===----------------------------------------------------------------------===//

#include "VPlanHCFGBuilder.h"
#include "LoopVectorizationPlanner.h"
#include "llvm/Analysis/LoopIterator.h"

#define DEBUG_TYPE "loop-vectorize"

using namespace llvm;

namespace {
// Class that is used to build the plain CFG for the incoming IR.
class PlainCFGBuilder {
private:
  // The outermost loop of the input loop nest considered for vectorization.
  Loop *TheLoop;

  // Loop Info analysis.
  LoopInfo *LI;

  // Vectorization plan that we are working on.
  VPlan &Plan;

  // Builder of the VPlan instruction-level representation.
  VPBuilder VPIRBuilder;

  // NOTE: The following maps are intentionally destroyed after the plain CFG
  // construction because subsequent VPlan-to-VPlan transformation may
  // invalidate them.
  // Map incoming BasicBlocks to their newly-created VPBasicBlocks and
  // VPRegionBlocks.
  DenseMap<BasicBlock *, VPBlockBase *> BB2VPB;
  // Map incoming Value definitions to their newly-created VPValues.
  DenseMap<Value *, VPValue *> IRDef2VPValue;

  // Hold phi node's that need to be fixed once the plain CFG has been built.
  SmallVector<PHINode *, 8> PhisToFix;

  /// Maps loops in the original IR to their corresponding region.
  DenseMap<Loop *, VPRegionBlock *> Loop2Region;

  // Utility functions.
  void setVPBBPredsFromBB(VPBlockBase *VPBB, BasicBlock *BB);
  void fixPhiNodes();
  VPBlockBase *getOrCreateVPB(BasicBlock *BB);
  VPBasicBlock *getOrCreateVPBB(BasicBlock *BB);
#ifndef NDEBUG
  bool isExternalDef(Value *Val);
#endif
  VPValue *getOrCreateVPOperand(Value *IRVal);
  void createVPInstructionsForVPBB(VPBasicBlock *VPBB, BasicBlock *BB);

public:
  PlainCFGBuilder(Loop *Lp, LoopInfo *LI, VPlan &P)
      : TheLoop(Lp), LI(LI), Plan(P) {}

  /// Build plain CFG for TheLoop  and connects it to Plan's entry.
  void buildPlainCFG();
};
} // anonymous namespace

// Set predecessors of \p VPBB in the same order as they are in \p BB. \p VPBB
// must have no predecessors.
void PlainCFGBuilder::setVPBBPredsFromBB(VPBlockBase *VPBB, BasicBlock *BB) {
  // Block already has predecessors set, skip it.
  if (VPBB->getNumPredecessors() != 0)
    return;

  SmallVector<VPBlockBase *, 8> VPBBPreds;
  // If VPBB is a region block, we are dealing with a loop header block. Connect
  // the region to the loop preheader, identified by checking the number of
  // successors (the latch won't have any successors yet, as it has not been
  // processed at this time.
  if (isa<VPRegionBlock>(VPBB)) {
    for (BasicBlock *Pred : predecessors(BB)) {
      auto *PredVPBB = getOrCreateVPBB(Pred);
      if (PredVPBB->getNumSuccessors() > 0)
        VPBBPreds.push_back(PredVPBB);
    }

    VPBB->setPredecessors(VPBBPreds);
    return;
  }

  // Collect VPBB predecessors.
  for (BasicBlock *Pred : predecessors(BB)) {
    auto *PredVPBB = getOrCreateVPBB(Pred);
    if (PredVPBB->getParent() == VPBB->getParent())
      VPBBPreds.push_back(PredVPBB);
    else
      VPBBPreds.push_back(PredVPBB->getParent());
  }

  VPBB->setPredecessors(VPBBPreds);
}

// Add operands to VPInstructions representing phi nodes from the input IR.
void PlainCFGBuilder::fixPhiNodes() {
  for (auto *Phi : PhisToFix) {
    assert(IRDef2VPValue.count(Phi) && "Missing VPInstruction for PHINode.");
    VPValue *VPVal = IRDef2VPValue[Phi];
    assert(isa<VPWidenPHIRecipe>(VPVal) &&
           "Expected WidenPHIRecipe for phi node.");
    auto *VPPhi = cast<VPWidenPHIRecipe>(VPVal);
    assert(VPPhi->getNumOperands() == 0 &&
           "Expected VPInstruction with no operands.");

    for (unsigned I = 0; I != Phi->getNumOperands(); ++I)
      VPPhi->addIncoming(getOrCreateVPOperand(Phi->getIncomingValue(I)),
                         getOrCreateVPBB(Phi->getIncomingBlock(I)));
  }
}

VPBlockBase *PlainCFGBuilder::getOrCreateVPB(BasicBlock *BB) {
  auto BlockIt = BB2VPB.find(BB);
  if (BlockIt != BB2VPB.end())
    // Retrieve existing VPB.
    return BlockIt->second;

  // Get or create a region for the loop containing BB.
  Loop *CurrentLoop = LI->getLoopFor(BB);
  VPRegionBlock *ParentR = nullptr;
  if (CurrentLoop) {
    auto Iter = Loop2Region.insert({CurrentLoop, nullptr});
    if (Iter.second)
      Iter.first->second = new VPRegionBlock(
          CurrentLoop->getHeader()->getName().str(), false /*isReplicator*/);
    ParentR = Iter.first->second;
  }

  // Create new VPBB.
  LLVM_DEBUG(dbgs() << "Creating VPBasicBlock for " << BB->getName() << "\n");
  VPBasicBlock *VPBB = new VPBasicBlock(BB->getName());
  VPBB->setParent(ParentR);
  if (CurrentLoop && CurrentLoop->getHeader() == BB) {
    BB2VPB[BB] = ParentR;
    ParentR->setEntry(VPBB);
    return ParentR;
  }
  BB2VPB[BB] = VPBB;
  return VPBB;
}

// Create a new empty VPBasicBlock for an incoming BasicBlock in the region
// corresponding to the containing loop  or retrieve an existing one if it was
// already created. If no region exists yet for the loop containing \p BB, a new
// one is created.
VPBasicBlock *PlainCFGBuilder::getOrCreateVPBB(BasicBlock *BB) {
  auto *VPB = getOrCreateVPB(BB);
  if (auto *Region = dyn_cast<VPRegionBlock>(VPB))
    return VPB->getEntryBasicBlock();
  return cast<VPBasicBlock>(VPB);
}

#ifndef NDEBUG
// Return true if \p Val is considered an external definition. An external
// definition is either:
// 1. A Value that is not an Instruction. This will be refined in the future.
// 2. An Instruction that is outside of the CFG snippet represented in VPlan,
// i.e., is not part of: a) the loop nest, b) outermost loop PH and, c)
// outermost loop exits.
bool PlainCFGBuilder::isExternalDef(Value *Val) {
  // All the Values that are not Instructions are considered external
  // definitions for now.
  Instruction *Inst = dyn_cast<Instruction>(Val);
  if (!Inst)
    return true;

  BasicBlock *InstParent = Inst->getParent();
  assert(InstParent && "Expected instruction parent.");

  // Check whether Instruction definition is in loop PH.
  BasicBlock *PH = TheLoop->getLoopPreheader();
  assert(PH && "Expected loop pre-header.");

  if (InstParent == PH)
    // Instruction definition is in outermost loop PH.
    return false;

  // Check whether Instruction definition is in the loop exit.
  BasicBlock *Exit = TheLoop->getUniqueExitBlock();
  assert(Exit && "Expected loop with single exit.");
  if (InstParent == Exit) {
    // Instruction definition is in outermost loop exit.
    return false;
  }

  // Check whether Instruction definition is in loop body.
  return !TheLoop->contains(Inst);
}
#endif

// Create a new VPValue or retrieve an existing one for the Instruction's
// operand \p IRVal. This function must only be used to create/retrieve VPValues
// for *Instruction's operands* and not to create regular VPInstruction's. For
// the latter, please, look at 'createVPInstructionsForVPBB'.
VPValue *PlainCFGBuilder::getOrCreateVPOperand(Value *IRVal) {
  auto VPValIt = IRDef2VPValue.find(IRVal);
  if (VPValIt != IRDef2VPValue.end())
    // Operand has an associated VPInstruction or VPValue that was previously
    // created.
    return VPValIt->second;

  // Operand doesn't have a previously created VPInstruction/VPValue. This
  // means that operand is:
  //   A) a definition external to VPlan,
  //   B) any other Value without specific representation in VPlan.
  // For now, we use VPValue to represent A and B and classify both as external
  // definitions. We may introduce specific VPValue subclasses for them in the
  // future.
  assert(isExternalDef(IRVal) && "Expected external definition as operand.");

  // A and B: Create VPValue and add it to the pool of external definitions and
  // to the Value->VPValue map.
  VPValue *NewVPVal = Plan.getVPValueOrAddLiveIn(IRVal);
  IRDef2VPValue[IRVal] = NewVPVal;
  return NewVPVal;
}

// Create new VPInstructions in a VPBasicBlock, given its BasicBlock
// counterpart. This function must be invoked in RPO so that the operands of a
// VPInstruction in \p BB have been visited before (except for Phi nodes).
void PlainCFGBuilder::createVPInstructionsForVPBB(VPBasicBlock *VPBB,
                                                  BasicBlock *BB) {
  VPIRBuilder.setInsertPoint(VPBB);
  for (Instruction &InstRef : *BB) {
    Instruction *Inst = &InstRef;

    // There shouldn't be any VPValue for Inst at this point. Otherwise, we
    // visited Inst when we shouldn't, breaking the RPO traversal order.
    assert(!IRDef2VPValue.count(Inst) &&
           "Instruction shouldn't have been visited.");

    if (auto *Br = dyn_cast<BranchInst>(Inst)) {
      // Conditional branch instruction are represented using BranchOnCond
      // recipes.
      if (Br->isConditional()) {
        VPValue *Cond = getOrCreateVPOperand(Br->getCondition());
        VPBB->appendRecipe(
            new VPInstruction(VPInstruction::BranchOnCond, {Cond}));
      }

      // Skip the rest of the Instruction processing for Branch instructions.
      continue;
    }

    VPValue *NewVPV;
    if (auto *Phi = dyn_cast<PHINode>(Inst)) {
      // Phi node's operands may have not been visited at this point. We create
      // an empty VPInstruction that we will fix once the whole plain CFG has
      // been built.
      NewVPV = new VPWidenPHIRecipe(Phi);
      VPBB->appendRecipe(cast<VPWidenPHIRecipe>(NewVPV));
      PhisToFix.push_back(Phi);
    } else {
      // Translate LLVM-IR operands into VPValue operands and set them in the
      // new VPInstruction.
      SmallVector<VPValue *, 4> VPOperands;
      for (Value *Op : Inst->operands())
        VPOperands.push_back(getOrCreateVPOperand(Op));

      // Build VPInstruction for any arbitrary Instruction without specific
      // representation in VPlan.
      NewVPV = cast<VPInstruction>(
          VPIRBuilder.createNaryOp(Inst->getOpcode(), VPOperands, Inst));
    }

    IRDef2VPValue[Inst] = NewVPV;
  }
}

// Main interface to build the plain CFG.
void PlainCFGBuilder::buildPlainCFG() {
  // 0. Reuse the top-level region, vector-preheader and exit VPBBs from the
  // skeleton.
  VPRegionBlock *TheRegion = Plan.getVectorLoopRegion();
  Loop2Region[TheLoop] = TheRegion;
  BasicBlock *ThePreheaderBB = TheLoop->getLoopPreheader();
  assert((ThePreheaderBB->getTerminator()->getNumSuccessors() == 1) &&
         "Unexpected loop preheader");
  VPBasicBlock *VectorPreheaderVPBB = Plan.getEntry();
  BB2VPB[ThePreheaderBB] = VectorPreheaderVPBB;
  BasicBlock *LoopExitBB = TheLoop->getUniqueExitBlock();
  assert(LoopExitBB && "Loops with multiple exits are not supported.");
  BB2VPB[LoopExitBB] = cast<VPBasicBlock>(TheRegion->getSingleSuccessor());

  // 1. Scan the body of the loop in a topological order to visit each basic
  // block after having visited its predecessor basic blocks. Create a VPBB for
  // each BB and link it to its successor and predecessor VPBBs. Note that
  // predecessors must be set in the same order as they are in the incomming IR.
  // Otherwise, there might be problems with existing phi nodes and algorithm
  // based on predecessors traversal.

  // Loop PH needs to be explicitly visited since it's not taken into account by
  // LoopBlocksDFS.
  VectorPreheaderVPBB->setName("vector.ph");
  for (auto &I : *ThePreheaderBB) {
    if (I.getType()->isVoidTy())
      continue;
    IRDef2VPValue[&I] = Plan.getVPValueOrAddLiveIn(&I);
  }
  // Create empty VPBB for header block of the top region and set its name.
  VPBlockBase *HeaderVPBB = getOrCreateVPBB(TheLoop->getHeader());
  HeaderVPBB->setName("vector.body");

  LoopBlocksRPO RPO(TheLoop);
  RPO.perform(LI);

  for (BasicBlock *BB : RPO) {
    // Create or retrieve the VPBasicBlock for this BB and create its
    // VPInstructions.
    VPBasicBlock *VPBB = getOrCreateVPBB(BB);
    createVPInstructionsForVPBB(VPBB, BB);

    // Set VPBB successors. We create empty VPBBs for successors if they don't
    // exist already. Recipes will be created when the successor is visited
    // during the RPO traversal.
    Instruction *TI = BB->getTerminator();
    assert(TI && "Terminator expected.");
    unsigned NumSuccs = TI->getNumSuccessors();

    if (NumSuccs == 1) {
      VPBlockBase *SuccVPBB = getOrCreateVPB(TI->getSuccessor(0));
      assert(SuccVPBB && "VPBB Successor not found.");
      VPBB->setOneSuccessor(SuccVPBB);
    } else if (NumSuccs == 2) {

      VPBlockBase *SuccVPBB0 = getOrCreateVPB(TI->getSuccessor(0));
      assert(SuccVPBB0 && "Successor 0 not found.");
      VPBlockBase *SuccVPBB1 = getOrCreateVPB(TI->getSuccessor(1));
      assert(SuccVPBB1 && "Successor 1 not found.");

      // Get VPBB's condition bit.
      assert(isa<BranchInst>(TI) && "Unsupported terminator!");
      // Look up the branch condition to get the corresponding VPValue
      // representing the condition bit in VPlan (which may be in another VPBB).
      assert(IRDef2VPValue.count(cast<BranchInst>(TI)->getCondition()) &&
             "Missing condition bit in IRDef2VPValue!");

      VPRegionBlock *Region = VPBB->getParent();
      // If VPBB is the loop latch i.e. one of its predecessor is the region
      // itself, update the region's exiting block and the successor of the
      // region.
      if (Region && (SuccVPBB0 == Region || SuccVPBB1 == Region)) {
        VPBlockBase *SuccVPBB = SuccVPBB0 == Region ? SuccVPBB1 : SuccVPBB0;
        if (Region->getNumSuccessors() == 0) {
          assert(SuccVPBB && "VPBB Successor not found.");
          Region->setOneSuccessor(SuccVPBB);
        }
        Region->setExiting(VPBB);
        Region->setParent(SuccVPBB->getParent());
      } else {
        // Link successors.
        VPBB->setTwoSuccessors(SuccVPBB0, SuccVPBB1);
      }
    } else
      llvm_unreachable("Number of successors not supported.");
    // Set VPBB predecessors in the same order as they are in the incoming BB.
    setVPBBPredsFromBB(getOrCreateVPB(BB), BB);
  }

  // 2. The whole CFG has been built at this point so all the input Values must
  // have a VPlan couterpart. Fix VPlan phi nodes by adding their corresponding
  // VPlan operands.
  fixPhiNodes();
}

void VPlanHCFGBuilder::buildPlainCFG() {
  PlainCFGBuilder PCFGBuilder(TheLoop, LI, Plan);
  PCFGBuilder.buildPlainCFG();
}

// Public interface to build a H-CFG.
void VPlanHCFGBuilder::buildHierarchicalCFG() {
  // Build Top Region enclosing the plain CFG.
  buildPlainCFG();
  LLVM_DEBUG(Plan.setName("HCFGBuilder: Plain CFG\n"); dbgs() << Plan);

  VPRegionBlock *TopRegion = Plan.getVectorLoopRegion();
  Verifier.verifyHierarchicalCFG(TopRegion);

  // Compute plain CFG dom tree for VPLInfo.
  VPDomTree.recalculate(Plan);
  LLVM_DEBUG(dbgs() << "Dominator Tree after building the plain CFG.\n";
             VPDomTree.print(dbgs()));
}
