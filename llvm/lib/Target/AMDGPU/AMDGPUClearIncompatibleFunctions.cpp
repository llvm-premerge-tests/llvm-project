//===-- AMDGPUClearIncompatibleFunctions.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file
/// This pass replaces the bodies of functions that have attributes incompatible
/// with the current target with trap/unreachable.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/Target/TargetMachine.h"
#include "llvm/IR/DiagnosticInfo.h"
#include "llvm/Pass.h"

#define DEBUG_TYPE "amdgpu-clear-incompatible-functions"

using namespace llvm;

namespace llvm {
extern const SubtargetFeatureKV AMDGPUFeatureKV[AMDGPU::NumSubtargetFeatures-1];
}

namespace {

using Generation = AMDGPUSubtarget::Generation;

class AMDGPUClearIncompatibleFunctions : public FunctionPass {
public:
  static char ID;

  AMDGPUClearIncompatibleFunctions(const TargetMachine *TM = nullptr) : FunctionPass(ID), TM(TM) {
    assert(TM && "No TargetMachine!");
  }

  StringRef getPassName() const override {
    return "AMDGPU Clear Incompatible Functions Bodies";
  }

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    // If changes are made, no analyses are preserved.
  }

  bool runOnFunction(Function &F) override;

private:
  const TargetMachine *TM = nullptr;
};

// List of features alongside the minimum GPU generation needed to support them.
constexpr std::array<std::pair<unsigned, Generation>, 6> FeatureAndMinGen = {{
  { AMDGPU::FeatureGFX11Insts, Generation::GFX11 },
  { AMDGPU::FeatureGFX10Insts, Generation::GFX10 },
  { AMDGPU::FeatureGFX9Insts, Generation::GFX9 },
  { AMDGPU::FeatureGFX8Insts, Generation::VOLCANIC_ISLANDS },
  { AMDGPU::FeatureDPP, Generation::VOLCANIC_ISLANDS },
  { AMDGPU::Feature16BitInsts, Generation::VOLCANIC_ISLANDS }
}};

StringRef GetFeatureName(unsigned Feature) {
  for (const SubtargetFeatureKV &KV : AMDGPUFeatureKV)
    if (Feature == KV.Value)
      return KV.Key;

  llvm_unreachable("Unknown Target feature");
}

} // end anonymous namespace

bool AMDGPUClearIncompatibleFunctions::runOnFunction(Function &F) {
  if (skipFunction(F) || F.empty())
    return false;

  LLVMContext &Ctx = F.getContext();
  const GCNSubtarget *ST = static_cast<const GCNSubtarget*>(TM->getSubtargetImpl(F));
  Generation GPUGen = ST->getGeneration();

  // Note: this pass checks attributes for GCN, so check we have a GCN GPU.
  if(GPUGen < Generation::SOUTHERN_ISLANDS)
    return false;

  bool Remove = false;
  for(const auto &[Feature, MinGPUGen]: FeatureAndMinGen) {
    if(ST->hasFeature(Feature) && GPUGen < MinGPUGen) {
      Remove = true;
      std::string Msg = "+" + GetFeatureName(Feature).str() + " is not supported on the current target. Deleting function body.";
      DiagnosticInfoUnsupported DiagInfo(F, Msg, DiagnosticLocation(), DS_Warning);
      Ctx.diagnose(DiagInfo);
    }
  }

  if (!Remove)
    return false;

  F.dropAllReferences();
  assert(F.empty());

  BasicBlock* Entry = BasicBlock::Create(Ctx, "entry", &F);
  IRBuilder<> Builder(Entry);
  Builder.CreateIntrinsic(Intrinsic::trap, {}, {});
  Builder.CreateUnreachable();
  return true;
}

INITIALIZE_PASS(AMDGPUClearIncompatibleFunctions, DEBUG_TYPE,
                "AMDGPU Clear Incompatible Functions Bodies", false, false)

char AMDGPUClearIncompatibleFunctions::ID = 0;

FunctionPass *llvm::createAMDGPUClearIncompatibleFunctionsPass(const TargetMachine *TM) {
  return new AMDGPUClearIncompatibleFunctions(TM);
}
