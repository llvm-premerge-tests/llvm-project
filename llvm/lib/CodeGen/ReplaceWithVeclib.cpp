//=== ReplaceWithVeclib.cpp - Replace vector intrinsics with veclib calls -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Replaces calls to LLVM vector intrinsics (i.e., calls to LLVM intrinsics
// with vector operands) with matching calls to functions from a vector
// library (e.g., libmvec, SVML) according to TargetLibraryInfo.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ReplaceWithVeclib.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/Analysis/DemandedBits.h"
#include "llvm/Analysis/GlobalsModRef.h"
#include "llvm/Analysis/OptimizationRemarkEmitter.h"
#include "llvm/Analysis/TargetLibraryInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InstIterator.h"
#include "llvm/Transforms/Utils/ModuleUtils.h"

using namespace llvm;

#define DEBUG_TYPE "replace-with-veclib"

STATISTIC(NumCallsReplaced,
          "Number of calls to intrinsics that have been replaced.");

STATISTIC(NumTLIFuncDeclAdded,
          "Number of vector library function declarations added.");

STATISTIC(NumFuncUsedAdded,
          "Number of functions added to `llvm.compiler.used`");

static bool
replaceWithTLIFunction(Instruction &I, const StringRef TLIName,
                       std::optional<ElementCount> NumElements = std::nullopt,
                       std::optional<bool> Masked = std::nullopt) {
  Module *M = I.getModule();
  CallInst *CI = dyn_cast<CallInst>(&I);

  // Check if the vector library function is already declared in this module,
  // otherwise insert it.
  Function *TLIFunc = M->getFunction(TLIName);
  StringRef OldName =
      CI ? CI->getCalledFunction()->getName() : I.getOpcodeName();
  if (!TLIFunc) {
    if (CI) {
      // Intrinsics handling.
      Function *OldFunc = CI->getCalledFunction();
      FunctionType *OldFuncTy = OldFunc->getFunctionType();
      TLIFunc =
          Function::Create(OldFuncTy, Function::ExternalLinkage, TLIName, *M);
      TLIFunc->copyAttributesFrom(OldFunc);
    } else {
      // FRem handling.
      assert(I.getOpcode() == Instruction::FRem &&
             "Must be a FRem instruction.");
      if (Masked && !NumElements)
        return false;
      Type *RetTy = I.getType();
      SmallVector<Type *> Tys = {RetTy, RetTy};
      if (Masked) {
        // Get the mask position.
        std::optional<llvm::VFInfo> Info =
            VFABI::tryDemangleForVFABI(TLIName, *M, NumElements.value());
        if (!Info)
          return false;
        std::optional<unsigned> MaskPos = Info->getParamIndexForOptionalMask();
        if (!MaskPos)
          return false;
        Tys.insert(Tys.begin() + MaskPos.value(),
                   VectorType::get(Type::getInt1Ty(M->getContext()),
                                   NumElements.value()));
      }
      TLIFunc = Function::Create(FunctionType::get(RetTy, Tys, false),
                                 Function::ExternalLinkage, TLIName, *M);
    }
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Added vector library function `"
                      << TLIName << "` of type `" << *(TLIFunc->getType())
                      << "` to module.\n");

    ++NumTLIFuncDeclAdded;

    // Add the freshly created function to llvm.compiler.used,
    // similar to as it is done in InjectTLIMappings
    appendToCompilerUsed(*M, {TLIFunc});

    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Adding `" << TLIName
                      << "` to `@llvm.compiler.used`.\n");
    ++NumFuncUsedAdded;
  }

  // Replace the call to the FRem instruction/vector intrinsic with a call
  // to the corresponding function from the vector library.
  IRBuilder<> IRBuilder(&I);
  CallInst *Replacement = nullptr;
  if (CI) {
    // Intrinsics handling.
    SmallVector<Value *> Args(CI->args());
    // Preserve the operand bundles.
    SmallVector<OperandBundleDef, 1> OpBundles;
    CI->getOperandBundlesAsDefs(OpBundles);
    Replacement = IRBuilder.CreateCall(TLIFunc, Args, OpBundles);
  } else {
    // FRem handling.
    if (Masked && !NumElements)
      return false;
    SmallVector<Value *> Args(I.operand_values());
    if (Masked)
      Args.push_back(IRBuilder.getAllOnesMask(NumElements.value()));
    Replacement = IRBuilder.CreateCall(TLIFunc, Args);
  }
  I.replaceAllUsesWith(Replacement);
  if (isa<FPMathOperator>(Replacement)) {
    // Preserve fast math flags for FP math.
    Replacement->copyFastMathFlags(&I);
  }
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Replaced call to `" << OldName
                    << "` with call to `" << TLIName << "`.\n");
  ++NumCallsReplaced;
  return true;
}

static bool replaceFremWithCallToVeclib(const TargetLibraryInfo &TLI,
                                        Instruction &I) {
  auto *VectorArgTy = dyn_cast<ScalableVectorType>(I.getType());
  // We have TLI mappings for FRem on scalable vectors only.
  if (!VectorArgTy)
    return false;
  ElementCount NumElements = VectorArgTy->getElementCount();
  auto *ElementType = VectorArgTy->getElementType();
  StringRef ScalarName;
  if (ElementType->isFloatTy())
    ScalarName = TLI.getName(LibFunc_fmodf);
  else if (ElementType->isDoubleTy())
    ScalarName = TLI.getName(LibFunc_fmod);
  else
    return false;
  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Looking up TLI mapping for `"
                    << ScalarName << "` and vector width " << NumElements
                    << ".\n");
  StringRef TLIName = TLI.getVectorizedFunction(ScalarName, NumElements);
  if (!TLIName.empty()) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Found unmasked TLI function `"
                      << TLIName << "`.\n");
    return replaceWithTLIFunction(I, TLIName);
  }
  TLIName = TLI.getVectorizedFunction(ScalarName, NumElements, true);
  if (!TLIName.empty()) {
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Found masked TLI function `"
                      << TLIName << "`.\n");
    return replaceWithTLIFunction(I, TLIName, NumElements, /*Masked*/ true);
  }
  return false;
}

static bool replaceWithCallToVeclib(const TargetLibraryInfo &TLI,
                                    CallInst &CI) {
  if (!CI.getCalledFunction()) {
    return false;
  }

  auto IntrinsicID = CI.getCalledFunction()->getIntrinsicID();
  if (IntrinsicID == Intrinsic::not_intrinsic) {
    // Replacement is only performed for intrinsic functions
    return false;
  }

  // Convert vector arguments to scalar type and check that
  // all vector operands have identical vector width.
  ElementCount VF = ElementCount::getFixed(0);
  SmallVector<Type *> ScalarTypes;
  for (auto Arg : enumerate(CI.args())) {
    auto *ArgType = Arg.value()->getType();
    // Vector calls to intrinsics can still have
    // scalar operands for specific arguments.
    if (isVectorIntrinsicWithScalarOpAtArg(IntrinsicID, Arg.index())) {
      ScalarTypes.push_back(ArgType);
    } else {
      // The argument in this place should be a vector if
      // this is a call to a vector intrinsic.
      auto *VectorArgTy = dyn_cast<VectorType>(ArgType);
      if (!VectorArgTy) {
        // The argument is not a vector, do not perform
        // the replacement.
        return false;
      }
      ElementCount NumElements = VectorArgTy->getElementCount();
      if (NumElements.isScalable()) {
        // The current implementation does not support
        // scalable vectors.
        return false;
      }
      if (VF.isNonZero() && VF != NumElements) {
        // The different arguments differ in vector size.
        return false;
      } else {
        VF = NumElements;
      }
      ScalarTypes.push_back(VectorArgTy->getElementType());
    }
  }

  // Try to reconstruct the name for the scalar version of this
  // intrinsic using the intrinsic ID and the argument types
  // converted to scalar above.
  std::string ScalarName;
  if (Intrinsic::isOverloaded(IntrinsicID)) {
    ScalarName = Intrinsic::getName(IntrinsicID, ScalarTypes, CI.getModule());
  } else {
    ScalarName = Intrinsic::getName(IntrinsicID).str();
  }

  if (!TLI.isFunctionVectorizable(ScalarName)) {
    // The TargetLibraryInfo does not contain a vectorized version of
    // the scalar function.
    return false;
  }

  // Try to find the mapping for the scalar version of this intrinsic
  // and the exact vector width of the call operands in the
  // TargetLibraryInfo.
  const std::string TLIName =
      std::string(TLI.getVectorizedFunction(ScalarName, VF));

  LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Looking up TLI mapping for `"
                    << ScalarName << "` and vector width " << VF << ".\n");

  if (!TLIName.empty()) {
    // Found the correct mapping in the TargetLibraryInfo,
    // replace the call to the intrinsic with a call to
    // the vector library function.
    LLVM_DEBUG(dbgs() << DEBUG_TYPE << ": Found TLI function `" << TLIName
                      << "`.\n");
    return replaceWithTLIFunction(CI, TLIName);
  }

  return false;
}

static bool runImpl(const TargetLibraryInfo &TLI, Function &F) {
  bool Changed = false;
  SmallVector<Instruction *> ReplacedCalls;
  for (auto &I : instructions(F)) {
    if (auto *CI = dyn_cast<CallInst>(&I)) {
      if (replaceWithCallToVeclib(TLI, *CI)) {
        ReplacedCalls.push_back(&I);
        Changed = true;
      }
    } else if (I.getOpcode() == Instruction::FRem) {
      // If there is a suitable TLI mapping for FRem instruction,
      // replace the instruction.
      if (replaceFremWithCallToVeclib(TLI, I)) {
        ReplacedCalls.push_back(&I);
        Changed = true;
      }
    }
  }
  // Erase the calls to the intrinsics and the FRem instructions that have been
  // replaced with calls to the vector library.
  for (auto *I : ReplacedCalls) {
    I->eraseFromParent();
  }
  return Changed;
}

////////////////////////////////////////////////////////////////////////////////
// New pass manager implementation.
////////////////////////////////////////////////////////////////////////////////
PreservedAnalyses ReplaceWithVeclib::run(Function &F,
                                         FunctionAnalysisManager &AM) {
  const TargetLibraryInfo &TLI = AM.getResult<TargetLibraryAnalysis>(F);
  auto Changed = runImpl(TLI, F);
  if (Changed) {
    PreservedAnalyses PA;
    PA.preserveSet<CFGAnalyses>();
    PA.preserve<TargetLibraryAnalysis>();
    PA.preserve<ScalarEvolutionAnalysis>();
    PA.preserve<LoopAccessAnalysis>();
    PA.preserve<DemandedBitsAnalysis>();
    PA.preserve<OptimizationRemarkEmitterAnalysis>();
    return PA;
  } else {
    // The pass did not replace any calls, hence it preserves all analyses.
    return PreservedAnalyses::all();
  }
}

////////////////////////////////////////////////////////////////////////////////
// Legacy PM Implementation.
////////////////////////////////////////////////////////////////////////////////
bool ReplaceWithVeclibLegacy::runOnFunction(Function &F) {
  const TargetLibraryInfo &TLI =
      getAnalysis<TargetLibraryInfoWrapperPass>().getTLI(F);
  return runImpl(TLI, F);
}

void ReplaceWithVeclibLegacy::getAnalysisUsage(AnalysisUsage &AU) const {
  AU.setPreservesCFG();
  AU.addRequired<TargetLibraryInfoWrapperPass>();
  AU.addPreserved<TargetLibraryInfoWrapperPass>();
  AU.addPreserved<ScalarEvolutionWrapperPass>();
  AU.addPreserved<AAResultsWrapperPass>();
  AU.addPreserved<OptimizationRemarkEmitterWrapperPass>();
  AU.addPreserved<GlobalsAAWrapperPass>();
}

////////////////////////////////////////////////////////////////////////////////
// Legacy Pass manager initialization
////////////////////////////////////////////////////////////////////////////////
char ReplaceWithVeclibLegacy::ID = 0;

INITIALIZE_PASS_BEGIN(ReplaceWithVeclibLegacy, DEBUG_TYPE,
                      "Replace intrinsics with calls to vector library", false,
                      false)
INITIALIZE_PASS_DEPENDENCY(TargetLibraryInfoWrapperPass)
INITIALIZE_PASS_END(ReplaceWithVeclibLegacy, DEBUG_TYPE,
                    "Replace intrinsics with calls to vector library", false,
                    false)

FunctionPass *llvm::createReplaceWithVeclibLegacyPass() {
  return new ReplaceWithVeclibLegacy();
}
