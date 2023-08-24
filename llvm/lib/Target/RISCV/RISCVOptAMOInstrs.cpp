//===- RISCVOptAMOInstrs.cpp - MI AMO instruction optimizations ----------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===---------------------------------------------------------------------===//
//
// This pass rewrites Rd to x0 for AMO instrs whose return values are unused.
//
//===---------------------------------------------------------------------===//

#include "RISCV.h"
#include "RISCVInstrInfo.h"
#include "RISCVSubtarget.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunctionPass.h"

using namespace llvm;
#define DEBUG_TYPE "riscv-opt-amo-instrs"
#define RISCV_OPT_AMO_INSTRS_NAME "RISC-V Optimize AMO Instructions"

STATISTIC(NumRewritedAMOInstrs, "Number of rewrited AMO instrs");

static cl::opt<bool>
    DisableAMORdDiscard("riscv-disable-amo-rd-discard",
                        cl::desc("Disable rewriting of rd for AMO instrs whose "
                                 "return values are unused"),
                        cl::init(false), cl::Hidden);

namespace {
class RISCVOptAMOInstrs : public MachineFunctionPass {
public:
  static char ID;

  RISCVOptAMOInstrs() : MachineFunctionPass(ID) {
    initializeRISCVOptAMOInstrsPass(*PassRegistry::getPassRegistry());
  }
  bool runOnMachineFunction(MachineFunction &MF) override;
  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.setPreservesCFG();
    MachineFunctionPass::getAnalysisUsage(AU);
  }

  StringRef getPassName() const override { return RISCV_OPT_AMO_INSTRS_NAME; }
};
} // end anonymous namespace

char RISCVOptAMOInstrs::ID = 0;
INITIALIZE_PASS(RISCVOptAMOInstrs, DEBUG_TYPE, RISCV_OPT_AMO_INSTRS_NAME, false,
                false)
char &llvm::RISCVOptAMOInstrsID = RISCVOptAMOInstrs::ID;

FunctionPass *llvm::createRISCVOptAMOInstrsPass() {
  return new RISCVOptAMOInstrs();
}

static bool isAMOInstr(const MachineInstr &MI) {
  switch (MI.getOpcode()) {
#define RISCV_HANDLE_AMO_INSTRS_FOR_EACH_ORDERING(NAME)                        \
  case RISCV::NAME:                                                            \
  case RISCV::NAME##_AQ:                                                       \
  case RISCV::NAME##_RL:                                                       \
  case RISCV::NAME##_AQ_RL:
#define RISCV_HANDLE_AMO_INSTRS(NAME)                                          \
  RISCV_HANDLE_AMO_INSTRS_FOR_EACH_ORDERING(NAME##_W)                          \
  RISCV_HANDLE_AMO_INSTRS_FOR_EACH_ORDERING(NAME##_D)
    RISCV_HANDLE_AMO_INSTRS(AMOSWAP)
    RISCV_HANDLE_AMO_INSTRS(AMOADD)
    RISCV_HANDLE_AMO_INSTRS(AMOAND)
    RISCV_HANDLE_AMO_INSTRS(AMOOR)
    RISCV_HANDLE_AMO_INSTRS(AMOXOR)
    RISCV_HANDLE_AMO_INSTRS(AMOMAX)
    RISCV_HANDLE_AMO_INSTRS(AMOMAXU)
    RISCV_HANDLE_AMO_INSTRS(AMOMIN)
    RISCV_HANDLE_AMO_INSTRS(AMOMINU)
#undef RISCV_HANDLE_AMO_INSTRS
#undef RISCV_HANDLE_AMO_INSTRS_FOR_EACH_ORDERING
    return true;
  default:
    return false;
  }
}

bool RISCVOptAMOInstrs::runOnMachineFunction(MachineFunction &MF) {
  if (DisableAMORdDiscard)
    return false;
  if (skipFunction(MF.getFunction()))
    return false;

  const RISCVSubtarget &ST = MF.getSubtarget<RISCVSubtarget>();
  if (!ST.hasStdExtA())
    return false;

  bool MadeChange = false;
  for (MachineBasicBlock &MBB : MF) {
    for (MachineInstr &MI : MBB) {
      if (!isAMOInstr(MI))
        continue;
      MachineOperand &DstReg = MI.getOperand(0);
      if (DstReg.isReg() && DstReg.isDead() && DstReg.getReg() != RISCV::X0) {
        DstReg.setReg(RISCV::X0);
        ++NumRewritedAMOInstrs;
        MadeChange = true;
      }
    }
  }

  return MadeChange;
}
