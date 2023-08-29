//===--------- SMEABI - SME  ABI-------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This pass implements parts of the the SME ABI, such as:
// * Using the lazy-save mechanism before enabling the use of ZA.
// * Setting up the lazy-save mechanism around invokes.
//
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "AArch64Subtarget.h"
#include "AArch64TargetMachine.h"
#include "Utils/AArch64BaseInfo.h"
#include "Utils/AArch64SMEAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/IntrinsicsAArch64.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/InitializePasses.h"
#include "llvm/Support/Debug.h"
#include "llvm/Transforms/Utils/Cloning.h"

using namespace llvm;

#define DEBUG_TYPE "aarch64-sme-abi"

namespace {
struct SMEABI : public FunctionPass {
  static char ID; // Pass identification, replacement for typeid
  SMEABI() : FunctionPass(ID) {
    initializeSMEABIPass(*PassRegistry::getPassRegistry());
  }

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
  }

private:
  bool updateNewZAFunctions(Module *M, Function *F, IRBuilder<> &Builder,
                            bool ClearZTState);
  bool handleExceptions(IRBuilder<> &Builder, Function *F, StructType *TPITy);
  AllocaInst *createZABufferAndTPIDR2Block(IRBuilder<> &Builder, Function *F,
                                           StructType *TPITy,
                                           StringRef ObjName);
  void setupLazySave(IRBuilder<> &Builder, Instruction *Call,
                     AllocaInst *TPIObj, BasicBlock *CheckBB,
                     BasicBlock *ResumeBB, BasicBlock *RestoreBB);
  void restoreLazySave(IRBuilder<> &Builder, Instruction *Call,
                       AllocaInst *TPIObj, BasicBlock *RestoreBB,
                       BasicBlock *ResumeBB = nullptr,
                       Value *PStateOnEntry = nullptr);
};
} // end anonymous namespace

char SMEABI::ID = 0;
static const char *name = "SME ABI Pass";
INITIALIZE_PASS_BEGIN(SMEABI, DEBUG_TYPE, name, false, false)
INITIALIZE_PASS_DEPENDENCY(TargetPassConfig)
INITIALIZE_PASS_END(SMEABI, DEBUG_TYPE, name, false, false)

FunctionPass *llvm::createSMEABIPass() { return new SMEABI(); }

//===----------------------------------------------------------------------===//
// Utility functions
//===----------------------------------------------------------------------===//

// Utility function to emit a call to __arm_sme_state and return 'pstate.sm' value.
Value *emitGetPStateSM(Module *M, IRBuilder<> &Builder) {
  auto *FTy = FunctionType::get(
      StructType::create({Builder.getInt64Ty(), Builder.getInt64Ty()}), {},
      /*IsVarArgs=*/false);
  auto Attrs =
      AttributeList()
          .addFnAttribute(M->getContext(), "aarch64_pstate_sm_compatible")
          .addFnAttribute(M->getContext(), "aarch64_pstate_za_preserved");
  FunctionCallee Callee = M->getOrInsertFunction("__arm_sme_state", FTy, Attrs);
  CallInst *Call = Builder.CreateCall(Callee);
  Call->setCallingConv(
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X2);

  // Extract PSTATE.SM from X0
  Value *X0 = Builder.CreateExtractValue(Call, 0);
  return Builder.CreateAnd(X0, Builder.getInt64(1));
}

// Utility function to emit a call to __arm_tpidr2_restore.
void emitTPIDR2Restore(Module *M, IRBuilder<> &Builder, Value *TPIDR2Obj) {
  auto *FTy = FunctionType::get(Builder.getVoidTy(), {Builder.getInt8PtrTy()},
                                /*IsVarArgs=*/false);
  auto Attrs =
      AttributeList()
          .addFnAttribute(M->getContext(), "aarch64_pstate_sm_compatible")
          .addFnAttribute(M->getContext(), "aarch64_pstate_za_shared");
  FunctionCallee Callee = M->getOrInsertFunction("__arm_tpidr2_restore", FTy, Attrs);
  CallInst *Call = Builder.CreateCall(
      Callee, Builder.CreatePointerCast(TPIDR2Obj, Builder.getInt8PtrTy()));
  Call->setCallingConv(
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0);
}

// Utility function to emit a call to __arm_tpidr2_save and clear TPIDR2_EL0.
void emitTPIDR2Save(Module *M, IRBuilder<> &Builder, bool ClearTPIDR2) {
  auto *TPIDR2SaveTy =
      FunctionType::get(Builder.getVoidTy(), {}, /*IsVarArgs=*/false);
  auto Attrs =
      AttributeList()
          .addFnAttribute(M->getContext(), "aarch64_pstate_sm_compatible")
          .addFnAttribute(M->getContext(), "aarch64_pstate_za_preserved");
  FunctionCallee Callee =
      M->getOrInsertFunction("__arm_tpidr2_save", TPIDR2SaveTy, Attrs);
  CallInst *Call = Builder.CreateCall(Callee);
  Call->setCallingConv(
      CallingConv::AArch64_SME_ABI_Support_Routines_PreserveMost_From_X0);

  if (ClearTPIDR2) {
    Function *WriteIntr =
        Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_set_tpidr2);
    Builder.CreateCall(WriteIntr->getFunctionType(), WriteIntr,
                       Builder.getInt64(0));
  }
}

void SMEABI::restoreLazySave(
    IRBuilder<> &Builder, Instruction *Call, AllocaInst *TPIObj,
    BasicBlock *RestoreBB, BasicBlock *ResumeBB, Value *PStateOnEntry) {
  Module *M = Call->getModule();

  // If Call is an Invoke, restore the lazy save in the normal destination.
  // Otherwise, restore the lazy save immediately after Call.
  if (auto *II = dyn_cast<InvokeInst>(Call))
    Builder.SetInsertPoint(II->getNormalDest()->getFirstNonPHIOrDbg());
  else
    Builder.SetInsertPoint(Call->getParent(), std::next(Call->getIterator()));

  // Re-enable pstate.za.
  Function *EnableZAIntr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_za_enable);
  Builder.CreateCall(EnableZAIntr->getFunctionType(), EnableZAIntr);

  // Create an intrinsic call to restore ZA, passing the 64-bit data pointer
  // to the TPIDR2 block.
  if (ResumeBB)
    Builder.SetInsertPoint(RestoreBB);
  emitTPIDR2Restore(M, Builder, TPIObj);
  if (ResumeBB)
    Builder.CreateBr(ResumeBB);
}

void SMEABI::setupLazySave(IRBuilder<> &Builder, Instruction *Call,
                           AllocaInst *TPIObj, BasicBlock *CheckBB,
                           BasicBlock *ResumeBB, BasicBlock *RestoreBB) {
  Module *M = Call->getModule();
  Builder.SetInsertPoint(Call);

  // Store the number of live slices to the num_za_save_slices field
  // of the TPIDR2 block
  Function *LiveIntr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_get_live_za_slices);
  auto *Live =
      Builder.CreateCall(LiveIntr->getFunctionType(), LiveIntr, {}, "live");
  auto *Trunc = Builder.CreateTrunc(Live, Builder.getInt16Ty(), "live.trunc");
  auto *TPILive = Builder.CreateGEP(TPIObj->getAllocatedType(), TPIObj,
                                    {Builder.getInt64(0), Builder.getInt32(1)},
                                    "tpidr2.obj.live");
  Builder.CreateStore(Trunc, TPILive);

  auto *PtrToInt =
      Builder.CreatePtrToInt(TPIObj, Builder.getInt64Ty(), "tpi.int");

  if (dyn_cast<InvokeInst>(Call)) {
    // Restart pstate.za if this is an Invoke, as we may be setting up
    // a lazy-save in the exception handler.
    // TODO: This will start pstate.za unnecessarily if the parent block is
    // not an unwind destination. It might be possible to improve this by
    // creating a mapping of blocks to ZA/SM states.
    Function *EnableZAIntr =
        Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_za_enable);
    Builder.CreateCall(EnableZAIntr->getFunctionType(), EnableZAIntr);
  }

  // Set TPIDR2_EL0 to the new object
  Function *WriteIntr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_set_tpidr2);
  Builder.CreateCall(WriteIntr->getFunctionType(), WriteIntr, PtrToInt);

  if (!CheckBB) {
    // If no CheckBB block was passed in, Call should be an Invoke with shared
    // or preserved ZA and we don't need to restore the lazy-save unless we
    // catch an exception. Abandon the lazy-save before resuming the function
    // at the normal destination.
    auto *Invoke = dyn_cast<InvokeInst>(Call);
    assert(Invoke && "CheckBB has not been provided for restoring lazy-save.");
    auto *FirstInst = Invoke->getNormalDest()->getFirstNonPHIOrDbg();
    auto *II = dyn_cast<IntrinsicInst>(FirstInst);
    // Avoid setting TPIDR2_EL0 to null more than once.
    if (!II || II->getIntrinsicID() != Intrinsic::aarch64_sme_set_tpidr2) {
      Builder.SetInsertPoint(FirstInst);
      Builder.CreateCall(WriteIntr->getFunctionType(), WriteIntr,
                         ConstantInt::get(Builder.getInt64Ty(), 0));
    }
    return;
  }

  // Check if the lazy save has been committed by the callee and should
  // be restored.
  Builder.SetInsertPoint(CheckBB);
  Function *GetEL0Intr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_get_tpidr2);
  auto *GetEL0 = Builder.CreateCall(GetEL0Intr->getFunctionType(), GetEL0Intr,
                                    {}, "tpidr2");
  auto *Cmp = Builder.CreateCmp(ICmpInst::ICMP_EQ, GetEL0,
                                ConstantInt::get(GetEL0->getType(), 0), "cmp");
  Builder.CreateCondBr(Cmp, RestoreBB, ResumeBB);

  // Conditionally store ZA after the call.
  restoreLazySave(Builder, Call, TPIObj, RestoreBB, ResumeBB);

  // Ensure we only set TPIDR2_EL0 to null once.
  auto *FirstInst = ResumeBB->getFirstNonPHIOrDbg();
  auto *II = dyn_cast<IntrinsicInst>(FirstInst);
  if (II && II->getIntrinsicID() == Intrinsic::aarch64_sme_set_tpidr2)
    return;

  // Set TPIDR2_EL0 to null before continuing with the rest of the function.
  // This will already be null if ZA was restored above, but is necessary
  // to abandon the lazy-save if the callee did not commit it.
  Builder.SetInsertPoint(FirstInst);
  Builder.CreateCall(WriteIntr->getFunctionType(), WriteIntr,
                     ConstantInt::get(Builder.getInt64Ty(), 0));

  return;
}

AllocaInst *SMEABI::createZABufferAndTPIDR2Block(
    IRBuilder<> &Builder, Function *F, StructType *TPITy, StringRef ObjName) {
  Module *M = F->getParent();

  Builder.SetInsertPoint(&*F->getEntryBlock().getFirstInsertionPt());
  auto *TPIObj = Builder.CreateAlloca(TPITy, nullptr, ObjName);

  // Allocate a buffer big enough to hold the max ZA size (SVL.B x SVL.B)
  Function *NIntr = Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_cntsb);
  auto *N = Builder.CreateCall(NIntr->getFunctionType(), NIntr, {}, "N");
  auto *NN = Builder.CreateMul(N, N, "NN");
  auto *Buffer = Builder.CreateAlloca(Builder.getInt8Ty(), NN, "buffer");
  Buffer->setAlignment(Align(16));

  // Fill the za_save_buffer field of the new TPIDR2 block
  auto *TPIBuffer = Builder.CreateGEP(
      TPIObj->getAllocatedType(), TPIObj,
      {Builder.getInt64(0), Builder.getInt32(0)}, ObjName + ".buffer");
  Builder.CreateStore(Buffer, TPIBuffer);

  // Clear reserved bytes 10-15 of the TPIDR2 block
  auto *Reserved = Builder.CreateAlloca(Builder.getIntNTy(48),
                                        Builder.getIntN(48, 0), "zero");
  Reserved->setAlignment(Align(16));
  TPIBuffer =
      Builder.CreateGEP(TPIObj->getAllocatedType(), TPIObj,
                        {Builder.getInt64(0), Builder.getInt32(2)}, "reserved");
  Builder.CreateStore(Reserved, TPIBuffer);

  return TPIObj;
}

bool SMEABI::handleExceptions(IRBuilder<> &Builder, Function *F,
                              StructType *TPITy) {
  Module *M = F->getParent();
  LLVMContext &Context = F->getContext();
  SmallVector<InvokeInst *, 4> Invokes;
  SmallVector<ResumeInst *, 4> Resumes;
  SmallVector<CallInst *, 4> BeginCatchCalls;
  SMEAttrs FnAttrs(*F);

  AllocaInst *TPIObj = nullptr;

   for (BasicBlock &BB : *F) {
    for (Instruction &I : BB) {
      if (auto *Invoke = dyn_cast<InvokeInst>(&I))
        Invokes.push_back(Invoke);
      else if (auto *Resume = dyn_cast<ResumeInst>(&I))
        Resumes.push_back(Resume);
      else if (auto *Call = dyn_cast<CallInst>(&I)) {
        if (Call->getCalledFunction() &&
            Call->getCalledFunction()->getName() == "__cxa_begin_catch")
          BeginCatchCalls.push_back(Call);
      }
    }
  }

  for (InvokeInst *Invoke : Invokes) {
    // We only need to set up and restore a lazy-save if there is ZA state.
    if (!FnAttrs.hasZAState())
      continue;

    if (!TPIObj)
      TPIObj =
          createZABufferAndTPIDR2Block(Builder, F, TPITy, "tpidr2.invoke.obj");

    SMEAttrs InvokeAttrs(*Invoke);
    if (InvokeAttrs.hasSharedZAInterface() || InvokeAttrs.preservesZA()) {
      // If the Invoke instruction is shared or preserved ZA, setupLazySave
      // does not need to restore the lazy-save at the normal destination.
      setupLazySave(Builder, Invoke, TPIObj, nullptr, nullptr, nullptr);
    } else {
      // Otherwise, set up new blocks for restoring ZA if the Invoke was
      // successful. Create a new block to read the value of TPIDR2 (CheckBB),
      // which becomes the new normal destination for the instruction.
      // Create another block to restore ZA (RestoreBB). ResumeBB is the
      // original NormalDest for the Invoke.
      auto *InvokeBB = Invoke->getParent();
      auto *ResumeBB = Invoke->getNormalDest();
      auto *CheckBB = BasicBlock::Create(Context, "check.za", F, ResumeBB);
      Invoke->setNormalDest(CheckBB);
      auto *RestoreBB = BasicBlock::Create(Context, "restore.za", F, ResumeBB);
      // Update any successor PHI nodes to match the new blocks.
      for (PHINode &PN : ResumeBB->phis()) {
        PN.replaceIncomingBlockWith(InvokeBB, CheckBB);
        PN.addIncoming(PN.getIncomingValueForBlock(CheckBB), RestoreBB);
      }
      ResumeBB->replaceSuccessorsPhiUsesWith(InvokeBB, ResumeBB);
      // Set-up the lazy save for this Invoke. This also handles restoring the
      // lazy save for the NormalDest.
      setupLazySave(Builder, Invoke, TPIObj, CheckBB, ResumeBB, RestoreBB);
    }

    if (!InvokeAttrs.hasNewZAInterface()) {
      // New ZA functions are the only function types which will commit
      // a lazy-save. For Invokes to any other type of function, create a
      // call which saves ZA to ensure that we restore the correct state
      // should the callee throw an unhandled exception that unwinds back
      // to the caller.
      Builder.SetInsertPoint(Invoke);
      emitTPIDR2Save(M, Builder, InvokeAttrs.hasSharedZAInterface());
    }
  }

  // We may need to restart streaming-mode depending on the starting value
  // of pstate.sm on entry to this function (if streaming-compatible).
  // Create a call to get this value in the entry block.
  Value *PStateOnEntry = ConstantInt::get(Builder.getInt64Ty(), 1);
  if (!Invokes.empty() && FnAttrs.hasStreamingCompatibleInterface()) {
    Builder.SetInsertPoint(&*F->getEntryBlock().getFirstInsertionPt());
    PStateOnEntry = emitGetPStateSM(M, Builder);
  }

  if (TPIObj) {
    // Ensure we stop pstate.za before calling the resume instruction.
    // TODO: We can improve this by not restoring the lazy-save & restarting
    // pstate.za in the first place before blocks which terminate in a resume.
    for (ResumeInst *Resume : Resumes) {
      Builder.SetInsertPoint(Resume);
      Function *DisableZAIntr =
          Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_za_disable);
      Builder.CreateCall(DisableZAIntr->getFunctionType(), DisableZAIntr);
    }
  }

  // Restore state at the beginning of each catch block.
  for (CallInst *Catch : BeginCatchCalls) {
    Catch->addFnAttr(Attribute::get(Context, "aarch64_pstate_za_preserved"));

    // Restore the lazy-save if there is ZA state.
    if (FnAttrs.hasZAState())
      restoreLazySave(Builder, Catch, TPIObj, Catch->getParent(),
                      /*ResumeBB*/ nullptr, PStateOnEntry);

    // Restart streaming-mode in the catch block if necessary.
    if (FnAttrs.requiresSMChange(SMEAttrs())) {
      auto *FirstInst = Catch->getParent()->getFirstNonPHIOrDbg();
      // Ensure that any catch blocks which are also landing pads keep the
      // LandingPadInst as the first instruction in the block.
      if (isa<LandingPadInst>(FirstInst))
        FirstInst = FirstInst->getNextNonDebugInstruction();
      Builder.SetInsertPoint(FirstInst);
      Function *EnableSMIntr = Intrinsic::getDeclaration(
          M, Intrinsic::aarch64_sme_invoke_resume_pstatesm);
      Builder.CreateCall(EnableSMIntr->getFunctionType(), EnableSMIntr,
                         PStateOnEntry);
    }
  }

  if (TPIObj || !FnAttrs.hasZAState()) {
    F->addFnAttr("aarch64_expanded_pstate_za");
    return true;
  }

  return false;
}

/// This function generates code to commit a lazy save at the beginning of a
/// function marked with `aarch64_pstate_za_new`. If the value read from
/// TPIDR2_EL0 is not null on entry to the function then the lazy-saving scheme
/// is active and we should call __arm_tpidr2_save to commit the lazy save.
/// Additionally, PSTATE.ZA should be enabled at the beginning of the function
/// and disabled before returning.
bool SMEABI::updateNewZAFunctions(Module *M, Function *F, IRBuilder<> &Builder,
                                  bool ClearZTState) {
  LLVMContext &Context = F->getContext();
  BasicBlock *OrigBB = &F->getEntryBlock();

  // Create the new blocks for reading TPIDR2_EL0 & enabling ZA state.
  auto *SaveBB = OrigBB->splitBasicBlock(OrigBB->begin(), "save.za", true);
  auto *PreludeBB = BasicBlock::Create(Context, "prelude", F, SaveBB);

  // Read TPIDR2_EL0 in PreludeBB & branch to SaveBB if not 0.
  Builder.SetInsertPoint(PreludeBB);
  Function *TPIDR2Intr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_get_tpidr2);
  auto *TPIDR2 = Builder.CreateCall(TPIDR2Intr->getFunctionType(), TPIDR2Intr,
                                    {}, "tpidr2");
  auto *Cmp =
      Builder.CreateCmp(ICmpInst::ICMP_NE, TPIDR2, Builder.getInt64(0), "cmp");
  Builder.CreateCondBr(Cmp, SaveBB, OrigBB);

  // Create a call __arm_tpidr2_save, which commits the lazy save.
  Builder.SetInsertPoint(&SaveBB->back());
  emitTPIDR2Save(M, Builder, /* Clear TPIDR2 */ true);

  // Enable pstate.za at the start of the function.
  Builder.SetInsertPoint(&OrigBB->front());
  Function *EnableZAIntr =
      Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_za_enable);
  Builder.CreateCall(EnableZAIntr->getFunctionType(), EnableZAIntr);

  // If SME2, clear ZT0 on entry to the function, after enabling pstate.za
  if (ClearZTState) {
    Function *ClearZT0Intr =
        Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_zero_zt);
    Builder.CreateCall(ClearZT0Intr->getFunctionType(), ClearZT0Intr,
                       {Builder.getInt32(0)});
  }

  // Before returning, disable pstate.za
  for (BasicBlock &BB : *F) {
    Instruction *T = BB.getTerminator();
    if (!T || !isa<ReturnInst>(T))
      continue;
    Builder.SetInsertPoint(T);
    Function *DisableZAIntr =
        Intrinsic::getDeclaration(M, Intrinsic::aarch64_sme_za_disable);
    Builder.CreateCall(DisableZAIntr->getFunctionType(), DisableZAIntr);
  }

  F->addFnAttr("aarch64_expanded_pstate_za");
  return true;
}

bool SMEABI::runOnFunction(Function &F) {
  Module *M = F.getParent();
  LLVMContext &Context = F.getContext();
  IRBuilder<> Builder(Context);

  TargetPassConfig &TPC = getAnalysis<TargetPassConfig>();

  StructType *TPITy =
      StructType::get(Context, {Builder.getInt8PtrTy(), Builder.getInt16Ty(),
                                ArrayType::get(Builder.getInt8Ty(), 6)});

  if (F.isDeclaration() || F.hasFnAttribute("aarch64_expanded_pstate_za"))
    return false;

  bool Changed = false;

  const AArch64Subtarget *Subtarget =
      TPC.getTM<AArch64TargetMachine>().getSubtargetImpl(F);

  SMEAttrs FnAttrs(F, Subtarget->hasSME2());
  if (FnAttrs.hasNewZAInterface())
    Changed |= updateNewZAFunctions(M, &F, Builder,
                                    FnAttrs.requiresPreservingZT(SMEAttrs()));

  if (FnAttrs.hasZAState() || FnAttrs.requiresSMChange(SMEAttrs()))
    Changed |= handleExceptions(Builder, &F, TPITy);

  return Changed;
}
