//===-- DesugarVariadicsPass.cpp --------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Can desugar variadic functions, their calls, va_arg and the intrinsics.
//
// The lowering replaces the variadic argument (...) with a Int8Ty*, moves
// function arguments into a alloca struct and passes that address instead.
// The struct alignment is the natural stack alignment, or greater if arguments
// have greater ABI alignment.
//
// Order of operations is chosen to keep the IR semantically well formed.
// 1/ Expand intrinsics that don't involve function parameters
// 2/ Declare new functions with the ... arg replaced with void*
// 3/ Replace call instructions to the variadics with calls to the declarations
// 4/ Splice the body of original functions into the new ones
// 5/ Delete the remaining parts of the original functions
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/DesugarVariadics.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/Cloning.h"

#include <cstdio>

#define DEBUG_TYPE "desugar-variadics"

using namespace llvm;

static cl::opt<bool>
    ApplyToAllOverride(DEBUG_TYPE "-all", cl::init(false),
                       cl::desc("Lower all variadic functions and calls"),
                       cl::Hidden);

namespace {

class DesugarVariadics : public ModulePass {
public:
  static char ID;
  bool ApplicableToAllDefault;
  DesugarVariadics(bool A = false)
      : ModulePass(ID), ApplicableToAllDefault(A) {}

  static void ExpandVAArg(VAArgInst *Inst, const DataLayout &DL) {
    auto &Ctx = Inst->getContext();

    Type *IntPtrTy = DL.getIntPtrType(Inst->getContext());
    IRBuilder<> Builder(Inst);

    Value *vaListPointer = Inst->getPointerOperand();
    Value *vaListValue = Builder.CreateLoad(Type::getInt8PtrTy(Ctx),
                                            vaListPointer, "arglist_current");

    Align DataAlign = DL.getABITypeAlign(Inst->getType());
    uint64_t DataAlignMinusOne = DataAlign.value() - 1;

    Value *Incr = Builder.CreateConstInBoundsGEP1_32(
        Type::getInt8Ty(Ctx), vaListValue, DataAlignMinusOne);

    Value *Mask = ConstantInt::get(IntPtrTy, ~(DataAlignMinusOne));
    Value *vaListAligned = Builder.CreateIntToPtr(
        Builder.CreateAnd(Builder.CreatePtrToInt(Incr, IntPtrTy), Mask),
        Incr->getType());

    auto *Result = Builder.CreateAlignedLoad(Inst->getType(), vaListAligned,
                                             DataAlign, "va_arg");
    Result->takeName(Inst);

    Value *Indexes[] = {ConstantInt::get(IntPtrTy, 1)};
    auto *Next = Builder.CreateInBoundsGEP(Inst->getType(), vaListAligned,
                                           Indexes, "arglist_next");
    Builder.CreateStore(Next, vaListPointer);

    Inst->replaceAllUsesWith(Result);
    Inst->eraseFromParent();
  }

  static void ExpandVAStart(VAStartInst *Inst, Argument *StructPtr) {
    IRBuilder<> Builder(Inst);
    Builder.CreateStore(StructPtr, Inst->getArgList());
    Inst->eraseFromParent();
  }

  static void ExpandVACopy(VACopyInst *Inst) {
    IRBuilder<> Builder(Inst);
    Value *dst = Inst->getDest();
    Value *src = Inst->getSrc();
    Value *ld = Builder.CreateLoad(src->getType(), src, "vacopy");
    Builder.CreateStore(ld, dst);
    Inst->eraseFromParent();
  }

  static void ExpandVAEnd(VAEndInst *Inst) { Inst->eraseFromParent(); }

  static bool runOnFunction(Function &F) {
    Module &M = *F.getParent();
    const DataLayout &DL = M.getDataLayout();
    bool Changed = false;
    for (BasicBlock &BB : F) {
      for (Instruction &I : llvm::make_early_inc_range(BB)) {
        if (VAArgInst *II = dyn_cast<VAArgInst>(&I)) {
          Changed = true;
          ExpandVAArg(II, DL);
          continue;
        }
        if (VAEndInst *II = dyn_cast<VAEndInst>(&I)) {
          Changed = true;
          ExpandVAEnd(II);
          continue;
        }
        if (VACopyInst *II = dyn_cast<VACopyInst>(&I)) {
          Changed = true;
          ExpandVACopy(II);
          continue;
        }
      }
    }

    if (F.isVarArg()) {
      ExpandVariadicFunction(M, F);
      Changed = true;
    }

    return Changed;
  }

  bool runOnModule(Module &M) override {
    bool Apply = ApplicableToAllDefault | ApplyToAllOverride;
    bool Changed = false;
    for (Function &F : llvm::make_early_inc_range(M))
      if (Apply || canTransformFunctionInIsolation(F))
        Changed |= runOnFunction(F);
    return Changed;
  }

  bool canTransformFunctionInIsolation(Function &F) {
    if (!F.isVarArg() || F.isDeclaration() || !F.hasLocalLinkage() ||
        F.hasAddressTaken() || F.hasFnAttribute(Attribute::Naked)) {
      return false;
    }

    // TODO: function is plumbing for extending this lowering pass for
    // optimisation on targets which use different variadic calling
    // conventions. Escape analysis on va_list values.
    return false;
  }

  static void ExpandCall(Module &M, CallBase *CB, Function *NF) {
    const DataLayout &DL = M.getDataLayout();

    FunctionType *FuncType = CB->getFunctionType();
    auto &Ctx = CB->getContext();
    unsigned NumArgs = FuncType->getNumParams();

    SmallVector<Value *> Args;
    Args.assign(CB->arg_begin(), CB->arg_begin() + NumArgs);

    SmallVector<std::pair<Value *, uint64_t>> Varargs;
    SmallVector<Type *> LocalVarTypes;

    Align MaxFieldAlign(1);
    uint64_t CurrentOffset = 0;
    for (unsigned I = FuncType->getNumParams(), E = CB->arg_size(); I < E;
         I++) {
      Value *ArgVal = CB->getArgOperand(I);

      bool isByVal = CB->paramHasAttr(I, Attribute::ByVal);
      if (isByVal)
        report_fatal_error("Unimplemented byval");

      Type *ArgType = ArgVal->getType();
      Align DataAlign = DL.getABITypeAlign(ArgType);
      MaxFieldAlign = std::max(MaxFieldAlign, DataAlign);
      uint64_t DataAlignV = DataAlign.value();

      if (uint64_t Rem = CurrentOffset % DataAlignV) {
        uint64_t Padding = DataAlignV - Rem;
        Type *ATy = ArrayType::get(Type::getInt8Ty(Ctx), Padding);
        LocalVarTypes.push_back(ATy);
        CurrentOffset += Padding;
      }

      Varargs.push_back({ArgVal, LocalVarTypes.size()});
      LocalVarTypes.push_back(ArgType);

      CurrentOffset += DL.getTypeAllocSize(ArgType).getFixedValue();
    }

    if (Varargs.empty()) {
      // todo, pass nullptr instead?
      LocalVarTypes.push_back(Type::getInt32Ty(Ctx));
    }

    StructType *VarargsTy = StructType::create(
        Ctx, LocalVarTypes, (Twine(NF->getName()) + ".vararg").str());

    Function *CBF = CB->getParent()->getParent();
    BasicBlock &BB = CBF->getEntryBlock();
    IRBuilder<> Builder(&*BB.getFirstInsertionPt());

    auto alloced = Builder.Insert(
        new AllocaInst(VarargsTy, DL.getAllocaAddrSpace(), nullptr,
                       std::max(MaxFieldAlign, assumedStructAlignment(DL))),
        "vararg_buffer");

    // TODO: Lifetime annotate it
    Builder.SetInsertPoint(CB);
    for (size_t i = 0; i < Varargs.size(); i++) {
      // todo: byval here?
      auto r = Builder.CreateStructGEP(VarargsTy, alloced, Varargs[i].second);
      Builder.CreateStore(Varargs[i].first,
                          r); // alignment info could be better
    }

    Args.push_back(Builder.CreatePointerBitCastOrAddrSpaceCast(
        alloced, Type::getInt8PtrTy(Ctx)));

    // Attributes excluding any on the vararg arguments
    AttributeList PAL = CB->getAttributes();
    if (!PAL.isEmpty()) {
      SmallVector<AttributeSet, 8> ArgAttrs;
      for (unsigned ArgNo = 0; ArgNo < NumArgs; ArgNo++)
        ArgAttrs.push_back(PAL.getParamAttrs(ArgNo));
      PAL = AttributeList::get(Ctx, PAL.getFnAttrs(), PAL.getRetAttrs(),
                               ArgAttrs);
    }

    SmallVector<OperandBundleDef, 1> OpBundles;
    CB->getOperandBundlesAsDefs(OpBundles);

    CallBase *NewCB = nullptr;
    if (InvokeInst *II = dyn_cast<InvokeInst>(CB)) {
      NewCB = InvokeInst::Create(NF, II->getNormalDest(), II->getUnwindDest(),
                                 Args, OpBundles, "", CB);
    } else {
      NewCB = CallInst::Create(NF, Args, OpBundles, "", CB);
      cast<CallInst>(NewCB)->setTailCallKind(
          cast<CallInst>(CB)->getTailCallKind());
    }

    NewCB->setAttributes(PAL);
    NewCB->takeName(CB);
    NewCB->setCallingConv(CB->getCallingConv());
    NewCB->copyMetadata(*CB, {LLVMContext::MD_prof, LLVMContext::MD_dbg});

    if (!CB->use_empty())
      CB->replaceAllUsesWith(NewCB);
    CB->eraseFromParent();
  }

  static Align assumedStructAlignment(const DataLayout &DL) {
    // TODO: Change the DataLayout API so there's an easier way to test
    // whether the stack alignment is known. Nvptx doesn't always have S
    // in the data layout string, this sidesteps an assertion there.
    Align ExcessiveAlignment = Align(UINT64_C(1) << 63u);
    bool knownNaturalStackAlignment =
        DL.exceedsNaturalStackAlignment(ExcessiveAlignment);
    if (knownNaturalStackAlignment) {
      return DL.getStackAlignment();
    } else {
      return {};
    }
  }

  static void ExpandVariadicFunction(Module &M, Function &F) {
    auto &Ctx = M.getContext();
    const DataLayout &DL = M.getDataLayout();

    IRBuilder<> Builder(Ctx);

    FunctionType *FTy = F.getFunctionType();

    SmallVector<Type *> ArgTypes(FTy->param_begin(), FTy->param_end());
    ArgTypes.push_back(Type::getInt8PtrTy(Ctx));

    FunctionType *NFTy =
        FunctionType::get(FTy->getReturnType(), ArgTypes, /*IsVarArgs*/ false);

    Function *NF = Function::Create(NFTy, F.getLinkage(), F.getAddressSpace());

    // Note - same attribute handling as DeadArgumentElimination
    NF->copyAttributesFrom(&F);
    NF->setComdat(F.getComdat());
    F.getParent()->getFunctionList().insert(F.getIterator(), NF);
    NF->takeName(&F);

    AttrBuilder ParamAttrs(Ctx);
    ParamAttrs.addAttribute(Attribute::NoAlias);
    ParamAttrs.addAlignmentAttr(assumedStructAlignment(DL));

    AttributeList Attrs = NF->getAttributes();
    Attrs = Attrs.addParamAttributes(Ctx, NFTy->getNumParams() - 1, ParamAttrs);
    NF->setAttributes(Attrs);

    // Declared the new function, can now create calls to it
    for (User *U : llvm::make_early_inc_range(F.users()))
      if (CallBase *CB = dyn_cast<CallBase>(U))
        ExpandCall(M, CB, NF);

    // If it's a definition, move the implementation across
    if (!F.isDeclaration()) {
      NF->splice(NF->begin(), &F);

      auto NewArg = NF->arg_begin();
      for (Argument &Arg : F.args()) {
        Arg.replaceAllUsesWith(NewArg);
        NewArg->takeName(&Arg);
        ++NewArg;
      }
      NewArg->setName("varargs");

      for (BasicBlock &BB : *NF)
        for (Instruction &I : llvm::make_early_inc_range(BB))
          if (VAStartInst *II = dyn_cast<VAStartInst>(&I))
            ExpandVAStart(II, NewArg);
    }

    SmallVector<std::pair<unsigned, MDNode *>, 1> MDs;
    F.getAllMetadata(MDs);
    for (auto [KindID, Node] : MDs)
      NF->addMetadata(KindID, *Node);

    // DAE bitcasts it, todo: check block addresses
    // This fails to update call instructions, unfortunately
    // It may therefore also fail to update globals
    F.replaceAllUsesWith(NF);

    F.eraseFromParent();
  }
};
} // namespace

char DesugarVariadics::ID = 0;

INITIALIZE_PASS(DesugarVariadics, DEBUG_TYPE, "Desugar Variadics", false,
                false)

ModulePass *llvm::createDesugarVariadicsPass(bool ApplicableToAllFunctions) {
  return new DesugarVariadics(ApplicableToAllFunctions);
}

PreservedAnalyses DesugarVariadicsPass::run(Module &M,
                                              ModuleAnalysisManager &) {
  return DesugarVariadics(false).runOnModule(M) ? PreservedAnalyses::none()
                                                  : PreservedAnalyses::all();
}
