//===-- AArch64PointerAuth.cpp - PtrAuth helpers --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "AArch64PointerAuth.h"

#include "AArch64BaseInfo.h"
#include "AArch64InstrInfo.h"
#include "AArch64Subtarget.h"

#include "llvm/CodeGen/MachineBasicBlock.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/MachineMemOperand.h"
#include "llvm/CodeGen/PseudoSourceValue.h"

using namespace llvm;

namespace {

class AddressCheckPseudoSourceValue : public PseudoSourceValue {
public:
  AddressCheckPseudoSourceValue(const TargetMachine &TM)
      : PseudoSourceValue(TargetCustom, TM) {}

  bool isConstant(const MachineFrameInfo *) const override { return false; }
  bool isAliased(const MachineFrameInfo *) const override { return true; }
  bool mayAlias(const MachineFrameInfo *) const override { return true; }
  void printCustom(raw_ostream &OS) const override { OS << "AddressCheck"; }
};

// Mark dummy LDR instruction as volatile to prevent removing it as dead code.
MachineMemOperand *createCheckMemOperand(MachineFunction &MF) {
  AddressCheckPseudoSourceValue CheckPSV(MF.getTarget());
  MachinePointerInfo PointerInfo(&CheckPSV);
  auto MOVolatileLoad =
      MachineMemOperand::MOLoad | MachineMemOperand::MOVolatile;

  return MF.getMachineMemOperand(PointerInfo, MOVolatileLoad, 4, Align(4));
}

} // namespace

MachineBasicBlock &llvm::checkAuthenticatedRegister(
    MachineBasicBlock::iterator MBBI, AuthCheckMethod Method,
    Register AuthenticatedReg, Register TmpReg, bool UseIKey, unsigned BrkImm) {

  MachineBasicBlock &MBB = *MBBI->getParent();
  MachineFunction &MF = *MBB.getParent();
  const AArch64Subtarget &Subtarget = MF.getSubtarget<AArch64Subtarget>();
  const AArch64InstrInfo *TII = Subtarget.getInstrInfo();
  DebugLoc DL = MBBI->getDebugLoc();

  // First, handle the methods not requiring creating extra MBBs.
  switch (Method) {
  default:
    break;
  case AuthCheckMethod::None:
    return MBB;
  case AuthCheckMethod::DummyLoad:
    BuildMI(MBB, MBBI, DL, TII->get(AArch64::LDRWui), getWRegFromXReg(TmpReg))
        .addReg(AArch64::LR)
        .addImm(0)
        .addMemOperand(createCheckMemOperand(MF))
        .setMIFlags(MachineInstr::FrameDestroy);
    return MBB;
  }

  // Control flow has to be changed, so arrange new MBBs.

  assert(MBBI != MBB.begin() &&
         "Cannot insert the check at the very beginning of MBB");
  // The block to insert check into.
  MachineBasicBlock *CheckBlock = &MBB;
  // The remaining part of the original MBB that is executed on success.
  MachineBasicBlock *SuccessBlock = MBB.splitAt(*std::prev(MBBI));

  // The block that explicitly generates a break-point exception on failure.
  MachineBasicBlock *BreakBlock =
      MF.CreateMachineBasicBlock(MBB.getBasicBlock());
  MF.push_back(BreakBlock);
  MBB.splitSuccessor(SuccessBlock, BreakBlock);

  assert(CheckBlock->getFallThrough() == SuccessBlock);
  BuildMI(BreakBlock, DL, TII->get(AArch64::BRK))
      .addImm(BrkImm)
      .setMIFlags(MachineInstr::FrameDestroy);

  switch (Method) {
  case AuthCheckMethod::None:
  case AuthCheckMethod::DummyLoad:
    llvm_unreachable("Should be handled above");
  case AuthCheckMethod::HighBitsNoTBI:
    BuildMI(CheckBlock, DL, TII->get(AArch64::EORXrs), TmpReg)
        .addReg(AuthenticatedReg)
        .addReg(AuthenticatedReg)
        .addImm(1)
        .setMIFlags(MachineInstr::FrameDestroy);
    BuildMI(CheckBlock, DL, TII->get(AArch64::TBNZX))
        .addReg(TmpReg)
        .addImm(62)
        .addMBB(BreakBlock)
        .setMIFlags(MachineInstr::FrameDestroy);
    return *SuccessBlock;
  case AuthCheckMethod::XPACHint:
    assert(AuthenticatedReg == AArch64::LR &&
           "XPACHint mode is only compatible with checking the LR register");
    assert(UseIKey && "XPACHint mode is only compatible with I-keys");
    BuildMI(CheckBlock, DL, TII->get(TargetOpcode::COPY), TmpReg)
        .addReg(AArch64::LR)
        .setMIFlags(MachineInstr::FrameDestroy);
    BuildMI(CheckBlock, DL, TII->get(AArch64::XPACLRI))
        .setMIFlags(MachineInstr::FrameDestroy);
    BuildMI(CheckBlock, DL, TII->get(AArch64::SUBSXrs), AArch64::XZR)
        .addReg(TmpReg)
        .addReg(AArch64::LR)
        .addImm(0)
        .setMIFlags(MachineInstr::FrameDestroy);
    BuildMI(CheckBlock, DL, TII->get(AArch64::Bcc))
        .addImm(AArch64CC::NE)
        .addMBB(BreakBlock)
        .setMIFlags(MachineInstr::FrameDestroy);
    return *SuccessBlock;
  }
}
