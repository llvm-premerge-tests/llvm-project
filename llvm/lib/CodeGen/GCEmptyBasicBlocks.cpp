#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineFunctionPass.h"
#include "llvm/CodeGen/MachineJumpTableInfo.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/InitializePasses.h"

using llvm::MachineJumpTableInfo;
using llvm::SmallVector;
using namespace llvm;

class GCEmptyBasicBlocks : public MachineFunctionPass {
public:
  static char ID;

  GCEmptyBasicBlocks() : MachineFunctionPass(ID) {
    initializeGCEmptyBasicBlocksPass(*PassRegistry::getPassRegistry());
  }

  StringRef getPassName() const override { return "Remove Basic Blocks."; }

  void getAnalysisUsage(AnalysisUsage &AU) const override;

  bool runOnMachineFunction(MachineFunction &MF) override;
};

bool GCEmptyBasicBlocks::runOnMachineFunction(MachineFunction &MF) {
  SmallVector<MachineBasicBlock *, 2> EmptyBlocks;
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
  if (EmptyBlocks.empty())
    return false;
  for (MachineBasicBlock *EMBB : EmptyBlocks) {
    auto NextMBBI = std::next(MachineFunction::iterator(EMBB));
    if (NextMBBI == MF.end())
      continue;
    for (auto &Pred : EMBB->predecessors())
      Pred->ReplaceUsesOfBlockWith(EMBB, &*NextMBBI);

    if (EMBB->isMachineBlockAddressTaken()) {
      for (auto &MBB : MF)
        MBB.ReplaceUsesOfBlockWith(EMBB, &*NextMBBI);
      if (JTI)
        JTI->ReplaceMBBInJumpTables(EMBB, &*NextMBBI);
    }
    EMBB->removeSuccessor(&*NextMBBI);
    EMBB->eraseFromParent();
  }
  return true;
}

char GCEmptyBasicBlocks::ID = 0;
INITIALIZE_PASS(GCEmptyBasicBlocks, "gc-empty-basic-blocks",
                "Removes empty basic blocks and redirects their uses to their "
                "fallthrough blocks.",
                false, false)

void GCEmptyBasicBlocks::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesAll();
  MachineFunctionPass::getAnalysisUsage(AU);
}

MachineFunctionPass *llvm::createGCEmptyBasicBlocksPass() {
  return new GCEmptyBasicBlocks();
}
