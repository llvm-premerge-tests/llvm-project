//===- EscapeEnumerator.cpp -----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Defines a helper class that enumerates all possible exits from a function,
// including exception handling.
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Utils/EscapeEnumerator.h"
#include "llvm/IR/EHPersonalities.h"
#include "llvm/IR/Module.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/Transforms/Utils/Local.h"

using namespace llvm;

static FunctionCallee getDefaultPersonalityFn(Module *M) {
  LLVMContext &C = M->getContext();
  Triple T(M->getTargetTriple());
  EHPersonality Pers = getDefaultEHPersonality(T);
  return M->getOrInsertFunction(getEHPersonalityName(Pers),
                                FunctionType::get(Type::getInt32Ty(C), true));
}

IRBuilder<> *EscapeEnumerator::Next() {
  if (Done)
    return nullptr;

  // Find all 'return', and 'resume' (but not 'resume unwindabort')
  // instructions; these instructions may transfer control to the calling
  // function.
  while (StateBB != StateE) {
    BasicBlock *CurBB = &*StateBB++;

    Instruction *TI = CurBB->getTerminator();
    if (!(isa<ReturnInst>(TI) ||
          (isa<ResumeInst>(TI) && !cast<ResumeInst>(TI)->isUnwindAbort())))
      continue;

    if (CallInst *CI = CurBB->getTerminatingMustTailCall())
      TI = CI;
    Builder.SetInsertPoint(TI);
    return &Builder;
  }

  Done = true;

  if (!HandleExceptions)
    return nullptr;

  if (F.doesNotThrow())
    return nullptr;

  // Find all 'call' instructions that may unwind to the calling function.
  // We cannot tranform calls with musttail tag, but it is handled above.
  SmallVector<Instruction *, 16> Calls;
  for (BasicBlock &BB : F)
    for (Instruction &II : BB)
      if (CallInst *CI = dyn_cast<CallInst>(&II))
        if (!CI->isUnwindAbort() && !CI->doesNotThrow() &&
            !CI->isMustTailCall())
          Calls.push_back(CI);

  if (Calls.empty())
    return nullptr;

  // Create a cleanup block.
  LLVMContext &C = F.getContext();
  BasicBlock *CleanupBB = BasicBlock::Create(C, CleanupBBName, &F);
  Type *ExnTy = StructType::get(Type::getInt8PtrTy(C), Type::getInt32Ty(C));
  if (!F.hasPersonalityFn()) {
    FunctionCallee PersFn = getDefaultPersonalityFn(F.getParent());
    F.setPersonalityFn(cast<Constant>(PersFn.getCallee()));
  }

  if (isScopedEHPersonality(classifyEHPersonality(F.getPersonalityFn()))) {
    report_fatal_error("Scoped EH not supported");
  }

  LandingPadInst *LPad =
      LandingPadInst::Create(ExnTy, 1, "cleanup.lpad", CleanupBB);
  LPad->setCleanup(true);
  ResumeInst *RI = ResumeInst::Create(LPad, CleanupBB);

  // Transform the 'call' instructions into 'invoke's branching to the
  // cleanup block. Go in reverse order to make prettier BB names.
  SmallVector<Value *, 16> Args;
  for (unsigned I = Calls.size(); I != 0;) {
    CallInst *CI = cast<CallInst>(Calls[--I]);
    changeToInvokeAndSplitBasicBlock(CI, CleanupBB, DTU);
  }

  Builder.SetInsertPoint(RI);
  return &Builder;
}
