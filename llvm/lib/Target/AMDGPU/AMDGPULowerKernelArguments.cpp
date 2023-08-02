//===-- AMDGPULowerKernelArguments.cpp ------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file This pass replaces accesses to kernel arguments with loads from
/// offsets from the kernarg base pointer.
//
//===----------------------------------------------------------------------===//

#include "AMDGPU.h"
#include "GCNSubtarget.h"
#include "SIMachineFunctionInfo.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/MDBuilder.h"
#include "llvm/IR/Metadata.h"
#include "llvm/Target/TargetMachine.h"
#define DEBUG_TYPE "amdgpu-lower-kernel-arguments"

using namespace llvm;

namespace {

class PreloadKernelArgInfo {
private:
  Function &F;

  const GCNSubtarget &ST;

  unsigned NumFreeUserSGPRs;

public:
  SmallVector<llvm::Metadata *, 8> KernelArgMetadata;

  PreloadKernelArgInfo(Function &F, const GCNSubtarget &ST) : F(F), ST(ST) {
    setInitialFreeUserSGPRsCount();
  }

  // Returns the maximum number of user SGPRs that we have available to preload
  // arguments.
  void setInitialFreeUserSGPRsCount() {
    const Module *M = F.getParent();

    const unsigned MaxUserSGRPs = 16;
    unsigned NumUserSGPRs = 0;

    const CallingConv::ID CC = F.getCallingConv();
    const bool IsKernel =
        CC == CallingConv::AMDGPU_KERNEL || CC == CallingConv::SPIR_KERNEL;
    const bool HasCalls = F.hasFnAttribute("amdgpu-calls");
    const bool HasStackObjects = F.hasFnAttribute("amdgpu-stack-objects");

    // ImplicitBufferPtr
    if (ST.isMesaGfxShader(F))
      NumUserSGPRs += 2;

    // PrivateSegmentBuffer
    if (ST.isAmdHsaOrMesa(F) && !ST.enableFlatScratch())
      NumUserSGPRs += 4;

    // DispatchPtr
    if (!F.hasFnAttribute("amdgpu-no-dispatch-ptr"))
      NumUserSGPRs += 2;

    // QueuePtr
    if (!F.hasFnAttribute("amdgpu-no-queue-ptr") &&
        AMDGPU::getCodeObjectVersion(*M) < AMDGPU::AMDHSA_COV5) {
      NumUserSGPRs += 2;
    }

    // KernargSegmentPtr
    if (IsKernel && (!F.arg_empty() || ST.getImplicitArgNumBytes(F) != 0))
      NumUserSGPRs += 2;

    // DispatchID
    if (!F.hasFnAttribute("amdgpu-no-dispatch-id"))
      NumUserSGPRs += 2;

    // FlatScratchInit
    if (ST.hasFlatAddressSpace() && AMDGPU::isEntryFunctionCC(CC) &&
        (ST.isAmdHsaOrMesa(F) || ST.enableFlatScratch()) &&
        (HasCalls || HasStackObjects || ST.enableFlatScratch()) &&
        !ST.flatScratchIsArchitected()) {
      NumUserSGPRs += 2;
    }

    // LDSKernelId
    if (!IsKernel && !F.hasFnAttribute("amdgpu-no-lds-kernel-id"))
      NumUserSGPRs += 1;

    NumFreeUserSGPRs = MaxUserSGRPs - NumUserSGPRs;
  }

  bool allocPreloadArg(unsigned AllocSize) {
    unsigned AllocSizeDword = alignTo(AllocSize, 4) / 4;
    if (AllocSizeDword > NumFreeUserSGPRs)
      return false;

    NumFreeUserSGPRs -= AllocSizeDword;
    return true;
  }
};

class AMDGPULowerKernelArguments : public FunctionPass{
public:
  static char ID;

  AMDGPULowerKernelArguments() : FunctionPass(ID) {}

  bool runOnFunction(Function &F) override;

  void getAnalysisUsage(AnalysisUsage &AU) const override {
    AU.addRequired<TargetPassConfig>();
    AU.setPreservesAll();
 }
};

} // end anonymous namespace

// skip allocas
static BasicBlock::iterator getInsertPt(BasicBlock &BB) {
  BasicBlock::iterator InsPt = BB.getFirstInsertionPt();
  for (BasicBlock::iterator E = BB.end(); InsPt != E; ++InsPt) {
    AllocaInst *AI = dyn_cast<AllocaInst>(&*InsPt);

    // If this is a dynamic alloca, the value may depend on the loaded kernargs,
    // so loads will need to be inserted before it.
    if (!AI || !AI->isStaticAlloca())
      break;
  }

  return InsPt;
}

bool AMDGPULowerKernelArguments::runOnFunction(Function &F) {
  CallingConv::ID CC = F.getCallingConv();
  if (CC != CallingConv::AMDGPU_KERNEL || F.arg_empty())
    return false;

  auto &TPC = getAnalysis<TargetPassConfig>();

  const TargetMachine &TM = TPC.getTM<TargetMachine>();
  const GCNSubtarget &ST = TM.getSubtarget<GCNSubtarget>(F);
  LLVMContext &Ctx = F.getParent()->getContext();
  const DataLayout &DL = F.getParent()->getDataLayout();
  BasicBlock &EntryBlock = *F.begin();
  IRBuilder<> Builder(&*getInsertPt(EntryBlock));

  const Align KernArgBaseAlign(16); // FIXME: Increase if necessary
  const uint64_t BaseOffset = ST.getExplicitKernelArgOffset();

  Align MaxAlign;
  // FIXME: Alignment is broken with explicit arg offset.;
  const uint64_t TotalKernArgSize = ST.getKernArgSegmentSize(F, MaxAlign);
  if (TotalKernArgSize == 0)
    return false;

  CallInst *KernArgSegment =
      Builder.CreateIntrinsic(Intrinsic::amdgcn_kernarg_segment_ptr, {}, {},
                              nullptr, F.getName() + ".kernarg.segment");

  KernArgSegment->addRetAttr(Attribute::NonNull);
  KernArgSegment->addRetAttr(
      Attribute::getWithDereferenceableBytes(Ctx, TotalKernArgSize));

  uint64_t ExplicitArgOffset = 0;
  // Preloaded kernel arguments must be sequential.
  bool InPreloadSequence = false;
  bool HasPreloadArgs = false;
  PreloadKernelArgInfo PreloadInfo(F, ST);

  int Idx = -1;
  for (Argument &Arg : F.args()) {
    Idx++;
    const bool IsByRef = Arg.hasByRefAttr();
    Type *ArgTy = IsByRef ? Arg.getParamByRefType() : Arg.getType();
    MaybeAlign ParamAlign = IsByRef ? Arg.getParamAlign() : std::nullopt;
    Align ABITypeAlign = DL.getValueOrABITypeAlignment(ParamAlign, ArgTy);

    uint64_t Size = DL.getTypeSizeInBits(ArgTy);
    uint64_t AllocSize = DL.getTypeAllocSize(ArgTy);

    uint64_t EltOffset = alignTo(ExplicitArgOffset, ABITypeAlign) + BaseOffset;
    ExplicitArgOffset = alignTo(ExplicitArgOffset, ABITypeAlign) + AllocSize;

    if (Arg.use_empty())
      continue;

    if (Arg.hasInRegAttr() && (InPreloadSequence || !HasPreloadArgs) &&
        PreloadInfo.allocPreloadArg(AllocSize)) {
      // Preload this argument.
      InPreloadSequence = true;
      HasPreloadArgs = true;
      MDBuilder MDB(Ctx);
      PreloadInfo.KernelArgMetadata.push_back(
          llvm::MDNode::get(Ctx, MDB.createConstant(llvm::ConstantInt::get(
                                     Builder.getInt32Ty(), Idx))));
    } else if (InPreloadSequence) {
      InPreloadSequence = false;
    }

    // If this is byval, the loads are already explicit in the function. We just
    // need to rewrite the pointer values.
    if (IsByRef) {
      Value *ArgOffsetPtr = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), KernArgSegment, EltOffset,
          Arg.getName() + ".byval.kernarg.offset");

      Value *CastOffsetPtr =
          Builder.CreateAddrSpaceCast(ArgOffsetPtr, Arg.getType());
      Arg.replaceAllUsesWith(CastOffsetPtr);
      continue;
    }

    if (PointerType *PT = dyn_cast<PointerType>(ArgTy)) {
      // FIXME: Hack. We rely on AssertZext to be able to fold DS addressing
      // modes on SI to know the high bits are 0 so pointer adds don't wrap. We
      // can't represent this with range metadata because it's only allowed for
      // integer types.
      if ((PT->getAddressSpace() == AMDGPUAS::LOCAL_ADDRESS ||
           PT->getAddressSpace() == AMDGPUAS::REGION_ADDRESS) &&
          !ST.hasUsableDSOffset())
        continue;

      // FIXME: We can replace this with equivalent alias.scope/noalias
      // metadata, but this appears to be a lot of work.
      if (Arg.hasNoAliasAttr())
        continue;
    }

    auto *VT = dyn_cast<FixedVectorType>(ArgTy);
    bool IsV3 = VT && VT->getNumElements() == 3;
    bool DoShiftOpt = Size < 32 && !ArgTy->isAggregateType();

    VectorType *V4Ty = nullptr;

    int64_t AlignDownOffset = alignDown(EltOffset, 4);
    int64_t OffsetDiff = EltOffset - AlignDownOffset;
    Align AdjustedAlign = commonAlignment(
        KernArgBaseAlign, DoShiftOpt ? AlignDownOffset : EltOffset);

    Value *ArgPtr;
    Type *AdjustedArgTy;
    if (DoShiftOpt) { // FIXME: Handle aggregate types
      // Since we don't have sub-dword scalar loads, avoid doing an extload by
      // loading earlier than the argument address, and extracting the relevant
      // bits.
      //
      // Additionally widen any sub-dword load to i32 even if suitably aligned,
      // so that CSE between different argument loads works easily.
      ArgPtr = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), KernArgSegment, AlignDownOffset,
          Arg.getName() + ".kernarg.offset.align.down");
      AdjustedArgTy = Builder.getInt32Ty();
    } else {
      ArgPtr = Builder.CreateConstInBoundsGEP1_64(
          Builder.getInt8Ty(), KernArgSegment, EltOffset,
          Arg.getName() + ".kernarg.offset");
      AdjustedArgTy = ArgTy;
    }

    if (IsV3 && Size >= 32) {
      V4Ty = FixedVectorType::get(VT->getElementType(), 4);
      // Use the hack that clang uses to avoid SelectionDAG ruining v3 loads
      AdjustedArgTy = V4Ty;
    }

    LoadInst *Load =
        Builder.CreateAlignedLoad(AdjustedArgTy, ArgPtr, AdjustedAlign);
    Load->setMetadata(LLVMContext::MD_invariant_load, MDNode::get(Ctx, {}));

    MDBuilder MDB(Ctx);

    if (isa<PointerType>(ArgTy)) {
      if (Arg.hasNonNullAttr())
        Load->setMetadata(LLVMContext::MD_nonnull, MDNode::get(Ctx, {}));

      uint64_t DerefBytes = Arg.getDereferenceableBytes();
      if (DerefBytes != 0) {
        Load->setMetadata(
          LLVMContext::MD_dereferenceable,
          MDNode::get(Ctx,
                      MDB.createConstant(
                        ConstantInt::get(Builder.getInt64Ty(), DerefBytes))));
      }

      uint64_t DerefOrNullBytes = Arg.getDereferenceableOrNullBytes();
      if (DerefOrNullBytes != 0) {
        Load->setMetadata(
          LLVMContext::MD_dereferenceable_or_null,
          MDNode::get(Ctx,
                      MDB.createConstant(ConstantInt::get(Builder.getInt64Ty(),
                                                          DerefOrNullBytes))));
      }

      if (MaybeAlign ParamAlign = Arg.getParamAlign()) {
        Load->setMetadata(
            LLVMContext::MD_align,
            MDNode::get(Ctx, MDB.createConstant(ConstantInt::get(
                                 Builder.getInt64Ty(), ParamAlign->value()))));
      }
    }

    // TODO: Convert noalias arg to !noalias

    if (DoShiftOpt) {
      Value *ExtractBits = OffsetDiff == 0 ?
        Load : Builder.CreateLShr(Load, OffsetDiff * 8);

      IntegerType *ArgIntTy = Builder.getIntNTy(Size);
      Value *Trunc = Builder.CreateTrunc(ExtractBits, ArgIntTy);
      Value *NewVal = Builder.CreateBitCast(Trunc, ArgTy,
                                            Arg.getName() + ".load");
      Arg.replaceAllUsesWith(NewVal);
    } else if (IsV3) {
      Value *Shuf = Builder.CreateShuffleVector(Load, ArrayRef<int>{0, 1, 2},
                                                Arg.getName() + ".load");
      Arg.replaceAllUsesWith(Shuf);
    } else {
      Load->setName(Arg.getName() + ".load");
      Arg.replaceAllUsesWith(Load);
    }
  }

  if (HasPreloadArgs) {
    F.setMetadata("preload_kernel_args",
                  llvm::MDNode::get(Ctx, PreloadInfo.KernelArgMetadata));
  }

  KernArgSegment->addRetAttr(
      Attribute::getWithAlignment(Ctx, std::max(KernArgBaseAlign, MaxAlign)));

  return true;
}

INITIALIZE_PASS_BEGIN(AMDGPULowerKernelArguments, DEBUG_TYPE,
                      "AMDGPU Lower Kernel Arguments", false, false)
INITIALIZE_PASS_END(AMDGPULowerKernelArguments, DEBUG_TYPE, "AMDGPU Lower Kernel Arguments",
                    false, false)

char AMDGPULowerKernelArguments::ID = 0;

FunctionPass *llvm::createAMDGPULowerKernelArgumentsPass() {
  return new AMDGPULowerKernelArguments();
}
