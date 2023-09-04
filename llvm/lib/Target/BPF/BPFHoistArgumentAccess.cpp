// ===------ BPFHoistArgumentAccess.cpp ------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// ===----------------------------------------------------------------------===//
//
// BPF verifier limits access patterns allowed for certain data
// types. E.g. struct __sk_buff and struct bpf_sock_ops. For these
// types only BASE + static-offset memory loads are allowed.
//
// This is so because offsets of the fields of these structures do not
// match real offsets in the running kernel. During BPF program
// load/verification loads and stores to the fields of these types are
// rewritten so that offsets match real offsets. For this rewrite to
// happen static offsets have to be encoded in the instructions.
//
// See kernel/bpf/verifier.c:convert_ctx_access function in the Linux
// kernel source tree for details.
//
// During instruction selection phase the following sequence of
// instructions:
//
//   %x = getelementptr %ptr, %imm
//   %y = load %x
//
// Is translated as a single load instruction with embedded offset,
// e.g. 'LDW %ptr, %imm', which matches access pattern necessary for
// the restricted set of types described above.
//
// However, several optimization passes might sink load/store instruction
// or hoist getelementptr instruction so that load/store pointer operand
// becomes a PHI.
//
// The BPFHoistArgumentAccess undoes sinking of the load / store
// instructions by hoisting load/store instructions to predecessor basic
// blocks:
// - transformation looks for loads or stores where base pointer is
//   defined via phi and some of the incoming values are getelementptr
//   such that:
//   - pointer operand is a function argument
//   - all indices are constant;
//   (alternatively, offset via preserve access index constructs is
//    also recognized)
// - if some predecessor basic block has multiple successors an
//   intermediate basic block is inserted between predecessor and phi's
//   basic block (thus guaranteeing that all predecessors have single
//   successor);
// - if some incoming values for predecessor basic block do not match the
//   getelementptr pattern a single intermediate basic block is is shared
//   by all these predecessors and phi's basic block;
// - load/store instruction and all instructions preceding it are hoisted
//   to predecessor basic blocks;
//
// For example:
//
// bb.then.1:                             bb.then.1:
//   %a = getelementptr                     br label %if.then.2
//   br label %if.then.2
//                                        bb.then.2:
// bb.then.2:                               %a1 = getelementptr
//   br label %if.end                       call void @foo
//                                          %l1 = load %a1
// bb.else:                         -->     br label %if.end
//   %b = call ptr @bar
//   br label %if.end                     bb.else:
//                                          %b = call ptr @bar
// if.end:                                  call void @foo
//   %p = phi [%a, label %bb.then.2],       %l2 = load %b
//            [%b, label %if.end]           br label %if.end
//   call void @foo
//   %l = load %p                         if.end:
//                                          %l = phi [%l1, label %bb.then.2],
//                                                   [%l2, label %if.end]

#include "BPF.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicsBPF.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"
#include "llvm/Transforms/Utils/ValueMapper.h"

#define DEBUG_TYPE "bpf-hoist-argument-access"

using namespace llvm;

[[noreturn]] static void reportUnexpected(Value *Value) {
  SmallString<64> Str;
  raw_svector_ostream Stream(Str);
  Stream << *Value;
  report_fatal_error(Twine("Unexpected value ").concat(Str));
}

using GEPChain = SmallVector<Instruction *, 2>;

GEPChain getHoistGEPChain(Value *Root) {
  GEPChain Chain;
  if (auto *GEP = dyn_cast<GetElementPtrInst>(Root)) {
    Chain.push_back(GEP);
  } else if (auto *Call = dyn_cast<CallInst>(Root)) {
    Chain.push_back(cast<Instruction>(Call->getArgOperand(1)));
    Chain.push_back(Call);
  } else {
    reportUnexpected(Root);
  }
  return Chain;
}

static bool isStaticArgumentAccess(Value *Insn) {
  if (auto *GEP = dyn_cast<GetElementPtrInst>(Insn)) {
    // Access to function argument via GEP with static offset.
    // Containing BB should have single successor.
    if (!GEP->hasAllConstantIndices())
      return false;
    if (!isa<Argument>(GEP->getPointerOperand()))
      return false;
  } else if (auto *Call = dyn_cast<CallInst>(Insn)) {
    // Access to function argument via preserve access index:
    //
    //   @"llvm.foo:0:20$0:3" = external global i64, !llvm.preserve.access.index !0
    //   ...
    //   %field.offset = load i64, ptr @"llvm.foo:0:20$0:3"
    //   %ptr.offset = getelementptr i8, ptr %argument, i64 %field.offset
    //   %ptr.result = call ptr @llvm.bpf.passthrough.p0.p0(i32 0, ptr %ptr.offset)
    //
    // Call BB should have a single successor, call and GEP should be
    // in the same BB.
    if (Call->getIntrinsicID() != Intrinsic::bpf_passthrough)
      return false;
    auto *GEP = dyn_cast<GetElementPtrInst>(Call->getArgOperand(1));
    if (!GEP)
      return false;
    if (GEP->getNumOperands() != 2)
      return false;
    if (!isa<Argument>(GEP->getPointerOperand()))
      return false;
    auto *OffsetLoad = dyn_cast<LoadInst>(GEP->getOperand(1));
    if (!OffsetLoad)
      return false;
    auto *GV = dyn_cast<GlobalVariable>(OffsetLoad->getPointerOperand());
    if (!GV)
      return false;
    if (!GV->getMetadata(LLVMContext::MD_preserve_access_index))
      return false;
  } else
    return false;
  return true;
}

static Value *getPointerOperand(Instruction *Insn) {
  if (auto *Load = dyn_cast<LoadInst>(Insn))
    return Load->getPointerOperand();
  if (auto *Store = dyn_cast<StoreInst>(Insn))
    return Store->getPointerOperand();
  return nullptr;
}

static Instruction *copyToBB(GEPChain &Chain, BasicBlock *BB) {
  Instruction *Clone = nullptr;
  IRBuilder<> Builder(BB->getTerminator());
  ValueToValueMapTy VMap;
  for (Instruction *I : Chain) {
    Clone = I->clone();
    VMap[I] = Clone;
    RemapInstruction(Clone, VMap, RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
    Builder.Insert(Clone, I->getName());
  }
  assert(Clone);
  return Clone;
}

static void eraseUnused(GEPChain &Chain) {
  for (Instruction *I : reverse(Chain))
    if (I->use_empty())
      I->eraseFromParent();
}

static void ensureSingleSuccessor(PHINode *Phi) {
  SmallVector<BasicBlock *> CatchAll;
  SmallVector<BasicBlock *> NeedSplit;
  for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I) {
    Value *Incoming = Phi->getIncomingValue(I);
    BasicBlock *BB = Phi->getIncomingBlock(I);
    if (isStaticArgumentAccess(Incoming)) {
      if (BB->getSingleSuccessor())
        continue;
      NeedSplit.push_back(BB);
      continue;
    }
    CatchAll.push_back(BB);
  }
  for (BasicBlock *BB : NeedSplit)
    SplitBlockPredecessors(Phi->getParent(), {BB}, ".sa.lh");
  if (CatchAll.size() > 1 ||
      (CatchAll.size() == 1 && !CatchAll[0]->getSinglePredecessor()))
    SplitBlockPredecessors(Phi->getParent(), CatchAll, ".lh");
  for (unsigned I = 0; I < Phi->getNumIncomingValues(); ++I) {
    Value *Incoming = Phi->getIncomingValue(I);
    BasicBlock *BB = Phi->getIncomingBlock(I);
    if (!isStaticArgumentAccess(Incoming))
      continue;
    GEPChain Chain = getHoistGEPChain(Incoming);
    if (all_of(Chain, [&](Instruction *I) { return I->getParent() == BB; }))
      continue;
    Phi->setIncomingValue(I, copyToBB(Chain, BB));
    eraseUnused(Chain);
  }
}

static bool hasUsesAfter(Instruction *Insn, Instruction *After) {
  for (User *U: Insn->users()) {
    Instruction *UI = dyn_cast<Instruction>(U);
    if (!UI)
      continue;
    if (UI->getParent() != After->getParent())
      return true;
    if (After->comesBefore(UI))
      return true;
  }
  return false;
}

static void hoistHeadToPredecessors(PHINode *Phi, Instruction *UpTo) {
  ensureSingleSuccessor(Phi);
  BasicBlock *ThisBB = UpTo->getParent();
  unsigned NumPreds = Phi->getNumIncomingValues();
  // Prepare replacement PHI nodes for instructions that would be
  // hoisted to predecessor blocks:
  // - HoistMap[&Insn] == nullptr if &Insn is used only
  //   in the part of ThisBB that is being hoisted;
  // - HoistMap[&Insn] == PHINode otherwise.
  SmallDenseMap<Instruction *, PHINode *> HoistMap;
  SmallVector<Instruction *, 8> Hoisted;
  for (Instruction &Insn : *ThisBB) {
    if (isa<PHINode>(&Insn)) {
      if (!hasUsesAfter(&Insn, UpTo)) {
        HoistMap[&Insn] = nullptr;
        Hoisted.push_back(&Insn);
      }
    } else {
      HoistMap[&Insn] = hasUsesAfter(&Insn, UpTo)
        ? PHINode::Create(Insn.getType(), NumPreds, Insn.getName())
        : nullptr;
      Hoisted.push_back(&Insn);
    }
    if (&Insn == UpTo)
      break;
  }
  // For each predecessor BB, copy instructions [ThisBB->begin()..UpTo]
  // from ThisBB to predecessor BB.
  for (unsigned I = 0; I < NumPreds; ++I) {
    BasicBlock *PredBB = Phi->getIncomingBlock(I);
    ValueToValueMapTy PredVMap;
    IRBuilder<> Builder(PredBB->getTerminator());
    for (Instruction &Insn : *ThisBB) {
      if (auto *SrcPhi = dyn_cast<PHINode>(&Insn)) {
        PredVMap[SrcPhi] = SrcPhi->getIncomingValue(I);
        continue;
      }
      Instruction *Clone = Insn.clone();
      PredVMap[&Insn] = Clone;
      RemapInstruction(Clone, PredVMap, RF_IgnoreMissingLocals | RF_NoModuleLevelChanges);
      Builder.Insert(Clone, Clone->getName());
      // If &Insn is used in the remaining part of ThisBB
      // add incoming value for corresponding PHI.
      if (auto *NewPhi = HoistMap[&Insn])
        NewPhi->addIncoming(Clone, PredBB);
      if (&Insn == UpTo)
        break;
    }
  }
  // Replace cloned instructions with PHI nodes.
  IRBuilder<> Builder(ThisBB->getFirstNonPHI());
  for (Instruction *Insn: reverse(Hoisted)) {
    PHINode *ReplacingPHI = HoistMap[Insn];
    if (ReplacingPHI) {
      Insn->replaceAllUsesWith(ReplacingPHI);
      Builder.Insert(ReplacingPHI, Insn->getName());
    }
    Insn->eraseFromParent();
  }
}

static bool expandSelects(Function &F) {
  SmallVector<SelectInst *, 64> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &Insn : BB) {
      Value *Pointer = getPointerOperand(&Insn);
      if (!Pointer)
        continue;
      SelectInst *Select = dyn_cast<SelectInst>(Pointer);
      if (!Select)
        continue;
      if (isStaticArgumentAccess(Select->getTrueValue()) ||
          isStaticArgumentAccess(Select->getFalseValue()))
        Worklist.push_back(Select);
    }

  if (Worklist.empty())
    return false;

  for (SelectInst *Select : Worklist) {
    LLVM_DEBUG(dbgs() << "expanding " << *Select << "\n");
    StringRef BaseName = Select->getParent()->getName();
    BasicBlock *TrueBB = nullptr;
    BasicBlock *FalseBB = nullptr;
    SplitBlockAndInsertIfThenElse(Select->getCondition(), Select, &TrueBB, &FalseBB);
    PHINode *Phi = PHINode::Create(Select->getType(), 2, Select->getName(), Select);
    Phi->addIncoming(Select->getTrueValue(), TrueBB);
    Phi->addIncoming(Select->getFalseValue(), FalseBB);
    TrueBB->setName(Twine(BaseName).concat(".select.true"));
    FalseBB->setName(Twine(BaseName).concat(".select.false"));
    Phi->getParent()->setName(Twine(BaseName).concat(".select.tail"));
    Select->replaceAllUsesWith(Phi);
    Select->eraseFromParent();
  }

  return true;
}

static bool shouldHoist(Instruction *Insn) {
  Value *Pointer = getPointerOperand(Insn);
  if (!Pointer)
    return false;
  auto *Phi = dyn_cast<PHINode>(Pointer);
  if (!Phi || Phi->getParent() != Insn->getParent())
    return false;
  for (Use &Incoming : Phi->incoming_values())
    if (isStaticArgumentAccess(Incoming))
      return true;
  return false;
}

static bool hoistGetElementPtrLoads(Function &F) {
  SmallVector<Instruction *, 64> Worklist;
  for (BasicBlock &BB : F)
    for (Instruction &Insn : BB)
      if (shouldHoist(&Insn))
        Worklist.push_back(&Insn);
  LLVM_DEBUG(dbgs() << "Going to hoist " << Worklist.size()
                    << " loads/stores\n");
  if (Worklist.empty())
    return false;
  for (Instruction *Insn: Worklist) {
    LLVM_DEBUG(dbgs() << "Hoisting " << *Insn << "\n");
    Value *Pointer = getPointerOperand(Insn);
    if (!Pointer || !isa<PHINode>(Pointer))
      reportUnexpected(Insn);
    hoistHeadToPredecessors(cast<PHINode>(Pointer), Insn);
  }
  return true;
}

PreservedAnalyses BPFHoistArgumentAccessPass::run(Function &F,
                                                  FunctionAnalysisManager &AM) {
  bool Changed = false;
  Changed |= expandSelects(F);
  Changed |= hoistGetElementPtrLoads(F);
  return Changed ? PreservedAnalyses::none() : PreservedAnalyses::all();
}
