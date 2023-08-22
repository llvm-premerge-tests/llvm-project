//===- ConvergenceVerifier.cpp - Verify convergence control -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/MachineConvergenceVerifier.h"
#include "llvm/CodeGen/GlobalISel/GenericMachineInstrs.h"
#include "llvm/CodeGen/MachineDominators.h"
#include "llvm/CodeGen/MachineRegisterInfo.h"
#include "llvm/CodeGen/MachineSSAContext.h"
#include "llvm/IR/GenericConvergenceVerifierImpl.h"

using namespace llvm;

template <>
const MachineInstr *
GenericConvergenceVerifier<MachineSSAContext>::findAndCheckConvergenceTokenUsed(
    const MachineInstr &MI) {
  auto &MRI = Context.getFunction()->getRegInfo();
  const MachineInstr *TokenDef = nullptr;

  for (auto &MO : MI.uses()) {
    if (!MO.isReg())
      continue;

    const auto RegTy = MRI.getType(MO.getReg());
    if (RegTy != LLT::token())
      continue;

    // A token type operand is a convergence control token iff its unique
    // definition is a convergence control intrinsic. We can't really verify
    // that since the token type may have other implicit uses. Instead we use it
    // as a way to identify convergence control token operands.
    const auto *Def = MRI.getUniqueVRegDef(MO.getReg());
    if (!Def)
      continue;
    if (!isConvergenceControlIntrinsic(MachineSSAContext::getIntrinsicID(*Def)))
      continue;

    CheckOrNull(MI.isCall() || isa<GIntrinsic>(MI),
                "Convergence control tokens can only be used by call "
                "instructions or intrinsics.",
                {Context.print(MO.getReg()), Context.print(&MI)});

    CheckOrNull(MO.isImplicit(),
                "Convergence control tokens can only be used implicitly.",
                {Context.print(MO.getReg()), Context.print(&MI)});

    CheckOrNull(!TokenDef,
                "A call can use at most one convergence control token.",
                {Context.print(MO.getReg()), Context.print(&MI)});

    TokenDef = Def;
  }

  if (TokenDef)
    Tokens[&MI] = TokenDef;

  return TokenDef;
}

template <>
bool GenericConvergenceVerifier<MachineSSAContext>::isInsideConvergentFunction(
    const MachineInstr &MI) {
  // The class MachineFunction does not have any property to indicate whether it
  // is convergent. Trivially return true so that the check always passes.
  return true;
}
template <>
bool GenericConvergenceVerifier<MachineSSAContext>::isConvergent(
    const MachineInstr &MI) {
  return MI.isConvergent();
}

template class llvm::GenericConvergenceVerifier<MachineSSAContext>;
