//===- VPlanCostModel.h - VPlan-based Vectorizer Cost Model ---------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// VPlan-based cost model
///
//===----------------------------------------------------------------------===//

#include "llvm/ADT/TypeSwitch.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/Analysis/VectorUtils.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Operator.h"
#include "llvm/Support/Debug.h"

#include "VPlan.h"
#include "VPlanCFG.h"
#include "VPlanCostModel.h"
#include "VPlanValue.h"

using namespace llvm;

#define DEBUG_TYPE "vplan-cost-model"

namespace llvm {
InstructionCost VPlanCostModel::expectedCost(const VPlan &Plan, ElementCount VF,
                                             bool &IsVec) {
  InstructionCost VectorIterCost = 0;
  for (const VPBlockBase *Block : vp_depth_first_deep(Plan.getEntry()))
    VectorIterCost += getCost(Block, VF, IsVec);

  return VectorIterCost;
}

InstructionCost VPlanCostModel::getCost(const VPBlockBase *Block,
                                        ElementCount VF, bool &IsVec) {
  return TypeSwitch<const VPBlockBase *, InstructionCost>(Block)
      .Case<VPBasicBlock>([&](const VPBasicBlock *BBlock) {
        InstructionCost Cost = 0;
        for (const VPRecipeBase &Recipe : *BBlock)
          Cost += getCost(Block, VF, IsVec);
        return Cost;
      })
      .Default([&](const VPBlockBase *BBlock) -> InstructionCost { return 0; });
}

InstructionCost VPlanCostModel::getCost(const VPRecipeBase *Recipe,
                                        ElementCount VF, bool &IsVec) {
  Type *VectorTy = nullptr;
  InstructionCost Cost =
      TypeSwitch<const VPRecipeBase *, InstructionCost>(Recipe)
          .Case<VPWidenMemoryInstructionRecipe>(
              [&](const VPWidenMemoryInstructionRecipe *VPWMIR) {
                return getMemoryOpCost(VPWMIR, VF, VectorTy);
              })
          .Default([&](const VPRecipeBase *R) -> InstructionCost {
            if (!R->hasUnderlyingInstr()) {
              LLVM_DEBUG(
                  dbgs() << "VPlanCM: unsupported recipe ";
                  VPSlotTracker SlotTracker((Recipe->getParent())
                                                ? Recipe->getParent()->getPlan()
                                                : nullptr);
                  Recipe->print(dbgs(), Twine(), SlotTracker); dbgs() << '\n');
              return 0;
            }
            Instruction *I = const_cast<Instruction *>(R->getUnderlyingInstr());
            return getLegacyInstructionCost(I, VF);
          });

  LLVM_DEBUG(dbgs() << "VPlanCM: cost " << Cost << " for VF " << VF
                    << " for VPInstruction: ";
             VPSlotTracker SlotTracker((Recipe->getParent())
                                           ? Recipe->getParent()->getPlan()
                                           : nullptr);
             Recipe->print(dbgs(), Twine(), SlotTracker); dbgs() << '\n');
  if (VF.isVector() && VectorTy->isVectorTy()) {
    if (unsigned NumParts = TTI.getNumberOfParts(VectorTy)) {
      if (VF.isScalable())
        // <vscale x 1 x iN> is assumed to be profitable over iN because
        // scalable registers are a distinct register class from scalar ones.
        // If we ever find a target which wants to lower scalable vectors
        // back to scalars, we'll need to update this code to explicitly
        // ask TTI about the register class uses for each part.
        IsVec |= NumParts <= VF.getKnownMinValue();
      else
        IsVec |= NumParts < VF.getKnownMinValue();
    } else
      Cost = InstructionCost::getInvalid();
  }
  return Cost;
}

InstructionCost VPlanCostModel::getMemoryOpCost(const Instruction *I, Type *Ty,
                                                bool IsConsecutive,
                                                bool IsMasked, bool IsReverse) {
  const Align Alignment = getLoadStoreAlignment(const_cast<Instruction *>(I));
  const Value *Ptr = getLoadStorePointerOperand(I);
  unsigned AS = getLoadStoreAddressSpace(const_cast<Instruction *>(I));
  if (IsConsecutive) {
    InstructionCost Cost = 0;
    if (IsMasked) {
      Cost += TTI.getMaskedMemoryOpCost(I->getOpcode(), Ty, Alignment, AS,
                                        CostKind);
    } else {
      TTI::OperandValueInfo OpInfo = TTI::getOperandInfo(I->getOperand(0));
      Cost += TTI.getMemoryOpCost(I->getOpcode(), Ty, Alignment, AS, CostKind,
                                  OpInfo, I);
    }
    if (IsReverse)
      Cost +=
          TTI.getShuffleCost(TargetTransformInfo::SK_Reverse,
                             cast<VectorType>(Ty), std::nullopt, CostKind, 0);
    return Cost;
  }
  return TTI.getAddressComputationCost(Ty) +
         TTI.getGatherScatterOpCost(I->getOpcode(), Ty, Ptr, IsMasked,
                                    Alignment, CostKind, I);
}

InstructionCost
VPlanCostModel::getMemoryOpCost(const VPWidenMemoryInstructionRecipe *VPWMIR,
                                ElementCount VF, Type *&VectorTy) {
  Instruction *I = &VPWMIR->getIngredient();
  Type *ValTy = getLoadStoreType(I);
  ValTy = truncateToMinimalBitwidth(ValTy, I);
  const bool IsMasked = VPWMIR->getMask() != nullptr;
  VectorTy = VectorType::get(ValTy, VF);

  return getMemoryOpCost(I, VectorTy, VPWMIR->isConsecutive(), IsMasked,
                         VPWMIR->isReverse());
}

} // namespace llvm
