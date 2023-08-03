//===- lib/Target/AMDGPU/AMDGPUInlineAsmLowering.h - Inline Asm Lowering -*- C++ -*---===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file describes how to lower inline assembly calls.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_TARGET_AMDGPU_AMDGPUINLINEASMLOWERING_H
#define LLVM_LIB_TARGET_AMDGPU_AMDGPUINLINEASMLOWERING_H

#include "llvm/CodeGen/GlobalISel/InlineAsmLowering.h"


namespace llvm {

class AMDGPUTargetLowering;
class GCNSubtarget;

class AMDGPUInlineAsmLowering final : public InlineAsmLowering {

public:
  AMDGPUInlineAsmLowering(const AMDGPUTargetLowering &TLI);

  bool buildAnyextOrCopy(Register Dst, Register Src,
                              MachineIRBuilder &MIRBuilder) const override;
};
} // End of namespace llvm;
#endif