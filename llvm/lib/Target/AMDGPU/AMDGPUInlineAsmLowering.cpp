//===-- llvm/lib/Target/AMDGPU/AMDGPUCallLowering.cpp - Call lowering -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file implements how to lower inline assembly calls for GlobalISel.
///
//===----------------------------------------------------------------------===//

#include "AMDGPUInlineAsmLowering.h"
#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/CodeGen/GlobalISel/MachineIRBuilder.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"

#define DEBUG_TYPE "amdgpu-inline-asm-lowering"

using namespace llvm;

AMDGPUInlineAsmLowering::AMDGPUInlineAsmLowering(
    const AMDGPUTargetLowering &TLI)
    : InlineAsmLowering(&TLI) {}

bool AMDGPUInlineAsmLowering::buildAnyextOrCopy(
    Register Dst, Register Src, MachineIRBuilder &MIRBuilder) const {

  MachineRegisterInfo &MRI = MIRBuilder.getMF().getRegInfo();

  const SIRegisterInfo *TRI =
      static_cast<const SIRegisterInfo *>(MRI.getTargetRegisterInfo());

  auto SrcTy = MRI.getType(Src);
  if (!SrcTy.isValid()) {
    LLVM_DEBUG(dbgs() << "Source type for copy is not valid\n");
    return false;
  }
  unsigned SrcSize = TRI->getRegSizeInBits(Src, MRI);
  unsigned DstSize = TRI->getRegSizeInBits(Dst, MRI);

  if (DstSize < SrcSize) {
    LLVM_DEBUG(dbgs() << "Input can't fit in destination reg class\n");
    return false;
  }

  // Attempt to anyext small scalar sources.
  if (DstSize > SrcSize) {
    if (!SrcTy.isScalar()) {
      LLVM_DEBUG(dbgs() << "Can't extend non-scalar input to size of"
                           "destination register class\n");
      return false;
    }
    Src = MIRBuilder.buildAnyExt(LLT::scalar(DstSize), Src).getReg(0);
  }

  // If this is a scalar assignment, insert a readfirstlane just in case the value
  // ends up from a VGPR.

  if (TRI->isSGPRReg(MRI, Dst) ) {
    auto ToSGPR = MIRBuilder
                      .buildIntrinsic(Intrinsic::amdgcn_readfirstlane,
                                      {MRI.getType(Src)})
                      .addReg(Src);
    Src = ToSGPR.getReg(0);
  }

  MIRBuilder.buildCopy(Dst, Src);
  return true;
}