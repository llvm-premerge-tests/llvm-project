#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"

using namespace llvm;

#define DEBUG_TYPE "gc-empty-blocks"

STATISTIC(NumEmptyBlocksRemoved, "Number of empty blocks removed");

class GCEmptyBasicBlocks : public MachineFunctionPass {
public:
  static char ID;

  GCEmptyBasicBlocks() : MachineFunctionPass(ID) {
    initializeGCEmptyBasicBlocksPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override {
    return "Remove Empty Basic Blocks.";
  }

  bool runOnMachineFunction(MachineFunction &MF) override;
};

bool GCEmptyBasicBlocks::runOnMachineFunction(MachineFunction &MF) {
  SmallVector<MachineBasicBlock *> EmptyBlocks;
  MachineJumpTableInfo *JTI = MF.getJumpTableInfo();
  for (auto &MBB : MF) {
    bool HasAnyRealCode = false;
    for (auto &MI : MBB) {
      if (!MI.isPosition() && !MI.isImplicitDef() && !MI.isKill() &&
          !MI.isDebugInstr()) {
        HasAnyRealCode = true;
      }
    }
    if (!HasAnyRealCode)
      EmptyBlocks.push_back(&MBB);
  }

  int NumRemoved = 0;
  for (MachineBasicBlock *EMBB : EmptyBlocks) {
    auto NextMBBI = std::next(MachineFunction::iterator(EMBB));
    // If this is the last block of the function, we have no way for adjust
    // the uses of this block. But normally, we shouldn't have any branches
    // to this block.
    if (NextMBBI == MF.end())
      continue;
    MachineBasicBlock *NextMBB = &*NextMBBI;
    // Rewire the predecessors of this block to use the next block.
    for (auto &Pred : EMBB->predecessors())
      Pred->ReplaceUsesOfBlockWith(EMBB, NextMBB);

    // If this block has its address taken, we must inspect and update every
    // block in the function and the jump table.
    if (EMBB->isMachineBlockAddressTaken()) {
      for (auto &MBB : MF)
        MBB.ReplaceUsesOfBlockWith(EMBB, NextMBB);
      if (JTI)
        JTI->ReplaceMBBInJumpTables(EMBB, NextMBB);
    }
    EMBB->removeSuccessor(NextMBB);
    EMBB->eraseFromParent();
    ++NumRemoved;
  }
  NumEmptyBlocksRemoved += NumRemoved;
  return NumRemoved != 0;
}

char GCEmptyBasicBlocks::ID = 0;
INITIALIZE_PASS(GCEmptyBasicBlocks, "gc-empty-basic-blocks",
                "Removes empty basic blocks and redirects their uses to their "
                "fallthrough blocks.",
                false, false)

MachineFunctionPass *llvm::createGCEmptyBasicBlocksPass() {
  return new GCEmptyBasicBlocks();
}
