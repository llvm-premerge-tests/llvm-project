//===- PPCInstructionSelector.cpp --------------------------------*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file implements the targeting of the InstructionSelector class for
/// PowerPC.
//===----------------------------------------------------------------------===//

#include "PPCInstrInfo.h"
#include "PPCRegisterBankInfo.h"
#include "PPCSubtarget.h"
#include "PPCTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelector.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelectorImpl.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/IR/IntrinsicsPowerPC.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "ppc-gisel"

using namespace llvm;

namespace {

#define GET_GLOBALISEL_PREDICATE_BITSET
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATE_BITSET

class PPCInstructionSelector : public InstructionSelector {
public:
  PPCInstructionSelector(const PPCTargetMachine &TM, const PPCSubtarget &STI,
                         const PPCRegisterBankInfo &RBI);

  bool select(MachineInstr &I) override;
  static const char *getName() { return DEBUG_TYPE; }

private:
  /// tblgen generated 'select' implementation that is used as the initial
  /// selector for the patterns that do not require complex C++.
  bool selectImpl(MachineInstr &I, CodeGenCoverage &CoverageInfo) const;

  bool selectConst(MachineInstr &I, MachineBasicBlock &MBB,
                   MachineRegisterInfo &MRI) const;
  bool selectSExt(MachineInstr &I, MachineBasicBlock &MBB,
                  MachineRegisterInfo &MRI) const;

  const PPCInstrInfo &TII;
  const PPCRegisterInfo &TRI;
  const PPCRegisterBankInfo &RBI;

#define GET_GLOBALISEL_PREDICATES_DECL
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATES_DECL

#define GET_GLOBALISEL_TEMPORARIES_DECL
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_TEMPORARIES_DECL
};

} // end anonymous namespace

#define GET_GLOBALISEL_IMPL
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_IMPL

PPCInstructionSelector::PPCInstructionSelector(const PPCTargetMachine &TM,
                                               const PPCSubtarget &STI,
                                               const PPCRegisterBankInfo &RBI)
    : TII(*STI.getInstrInfo()), TRI(*STI.getRegisterInfo()), RBI(RBI),
#define GET_GLOBALISEL_PREDICATES_INIT
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_PREDICATES_INIT
#define GET_GLOBALISEL_TEMPORARIES_INIT
#include "PPCGenGlobalISel.inc"
#undef GET_GLOBALISEL_TEMPORARIES_INIT
{
}

static bool selectCopy(MachineInstr &I, const TargetInstrInfo &TII,
                       MachineRegisterInfo &MRI, const TargetRegisterInfo &TRI,
                       const RegisterBankInfo &RBI) {
  Register DstReg = I.getOperand(0).getReg();
  Register SrcReg = I.getOperand(1).getReg();

  if (!Register::isPhysicalRegister(DstReg))
    if (!RBI.constrainGenericRegister(DstReg, PPC::G8RCRegClass, MRI))
      return false;
  if (!Register::isPhysicalRegister(SrcReg))
    if (!RBI.constrainGenericRegister(SrcReg, PPC::G8RCRegClass, MRI))
      return false;

  return true;
}

bool PPCInstructionSelector::selectConst(MachineInstr &I,
                                         MachineBasicBlock &MBB,
                                         MachineRegisterInfo &MRI) const {
  assert(I.getOpcode() == TargetOpcode::G_CONSTANT && "Unexpected G code");

  MachineInstr *MI = nullptr;
  Register DstReg = I.getOperand(0).getReg();
  APInt ConstValue = I.getOperand(1).getCImm()->getValue();
  if (ConstValue.isIntN(16)) {
    bool NeedMask = !ConstValue.isIntN(15);
    uint64_t Cst = ConstValue.getZExtValue();
    Register TmpReg =
        NeedMask ? MRI.createVirtualRegister(&PPC::G8RCRegClass) : DstReg;
    MI =
        BuildMI(MBB, I, I.getDebugLoc(), TII.get(PPC::LI8), TmpReg).addImm(Cst);
    if (NeedMask) {
      constrainSelectedInstRegOperands(*MI, TII, TRI, RBI);
      MI = BuildMI(MBB, I, I.getDebugLoc(), TII.get(PPC::RLDIC), DstReg)
               .addReg(TmpReg, RegState::Kill)
               .addImm(0)
               .addImm(16);
    }
  } else if (ConstValue.isSignedIntN(16)) {
    int64_t Cst = ConstValue.getSExtValue();
    MI =
        BuildMI(MBB, I, I.getDebugLoc(), TII.get(PPC::LI8), DstReg).addImm(Cst);
  } else if (ConstValue.isSignedIntN(32)) {
    int64_t Cst = ConstValue.getSExtValue();
    int64_t UpperCst = Cst >> 16;
    int64_t LowerCst = Cst & 0xffff;
    Register TmpReg = MRI.createVirtualRegister(&PPC::G8RCRegClass);
    MI = BuildMI(MBB, I, I.getDebugLoc(), TII.get(PPC::LIS8), TmpReg)
             .addImm(UpperCst);
    constrainSelectedInstRegOperands(*MI, TII, TRI, RBI);
    MI = BuildMI(MBB, I, I.getDebugLoc(), TII.get(PPC::ORI8), DstReg)
             .addReg(TmpReg, RegState::Kill)
             .addImm(LowerCst);
  } else if (ConstValue.isIntN(32)) {
    bool NeedMask = !ConstValue.isIntN(31);
    uint64_t Cst = ConstValue.getZExtValue();
    uint64_t UpperCst = Cst >> 16;
    uint64_t LowerCst = Cst & 0xffff;
    Register TmpReg =
        NeedMask ? MRI.createVirtualRegister(&PPC::G8RCRegClass) : DstReg;
    if (UpperCst == 0xffff && (LowerCst & 0x8000) == 0x8000) {
      MI = BuildMI(MBB, I, I.getDebugLoc(), TII.get(PPC::LI8), TmpReg)
               .addImm(LowerCst);
    } else {
      Register Tmp2Reg = MRI.createVirtualRegister(&PPC::G8RCRegClass);
      MI = BuildMI(MBB, I, I.getDebugLoc(), TII.get(PPC::LIS8), Tmp2Reg)
               .addImm(UpperCst);
      constrainSelectedInstRegOperands(*MI, TII, TRI, RBI);
      MI = BuildMI(MBB, I, I.getDebugLoc(), TII.get(PPC::ORI8), TmpReg)
               .addReg(Tmp2Reg, RegState::Kill)
               .addImm(LowerCst);
    }
    if (NeedMask) {
      constrainSelectedInstRegOperands(*MI, TII, TRI, RBI);
      MI = BuildMI(MBB, I, I.getDebugLoc(), TII.get(PPC::RLDIC), DstReg)
               .addReg(TmpReg, RegState::Kill)
               .addImm(0)
               .addImm(32);
    }
  } else
    return false;
  I.eraseFromParent();
  return constrainSelectedInstRegOperands(*MI, TII, TRI, RBI);
}

bool PPCInstructionSelector::selectSExt(MachineInstr &I, MachineBasicBlock &MBB,
                                        MachineRegisterInfo &MRI) const {
  assert(I.getOpcode() == TargetOpcode::G_SEXT_INREG && "Unexpected G code");

  Register DstReg = I.getOperand(0).getReg();
  Register SrcReg = I.getOperand(1).getReg();

  unsigned Opc;
  switch (I.getOperand(2).getImm()) {
  case 8:
    Opc = PPC::EXTSB8;
    break;
  case 16:
    Opc = PPC::EXTSH8;
    break;
  case 32:
    Opc = PPC::EXTSW;
    break;
  default:
    return false;
  }
  MachineInstr *MI =
      BuildMI(MBB, I, I.getDebugLoc(), TII.get(Opc), DstReg).addReg(SrcReg);
  I.eraseFromParent();
  return constrainSelectedInstRegOperands(*MI, TII, TRI, RBI);
}

bool PPCInstructionSelector::select(MachineInstr &I) {
  auto &MBB = *I.getParent();
  auto &MF = *MBB.getParent();
  auto &MRI = MF.getRegInfo();

  if (!isPreISelGenericOpcode(I.getOpcode())) {
    if (I.isCopy())
      return selectCopy(I, TII, MRI, TRI, RBI);

    return true;
  }

  if (selectImpl(I, *CoverageInfo))
    return true;

  switch (I.getOpcode()) {
  case TargetOpcode::G_CONSTANT:
    return selectConst(I, MBB, MRI);
  case TargetOpcode::G_SEXT_INREG:
    return selectSExt(I, MBB, MRI);
  default:
    return false;
  }
}

namespace llvm {
InstructionSelector *
createPPCInstructionSelector(const PPCTargetMachine &TM,
                             const PPCSubtarget &Subtarget,
                             const PPCRegisterBankInfo &RBI) {
  return new PPCInstructionSelector(TM, Subtarget, RBI);
}
} // end namespace llvm
