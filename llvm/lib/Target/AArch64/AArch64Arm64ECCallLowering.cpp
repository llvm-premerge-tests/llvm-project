//===-- AArch64Arm64ECCallLowering.cpp - Lower Arm64EC calls ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file contains the IR transform to lower external or indirect calls for
/// the ARM64EC calling convention. Such calls must go through the runtime, so
/// we can translate the calling convention for calls into the emulator.
///
/// This subsumes Control Flow Guard handling.
///
//===----------------------------------------------------------------------===//

#include "AArch64.h"
#include "llvm/ADT/SmallString.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/IR/CallingConv.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/TargetParser/Triple.h"
#include "llvm/InitializePasses.h"
#include "llvm/Pass.h"

using namespace llvm;

using OperandBundleDef = OperandBundleDefT<Value *>;

#define DEBUG_TYPE "arm64eccalllowering"

STATISTIC(Arm64ECCallsLowered, "Number of Arm64EC calls lowered");

static cl::opt<bool> LowerDirectToIndirect(
    "arm64ec-lower-direct-to-indirect", cl::Hidden, cl::init(true));
static cl::opt<bool> GenerateThunks(
    "arm64ec-generate-thunks", cl::Hidden, cl::init(true));

namespace {

class AArch64Arm64ECCallLowering : public ModulePass {
public:
  static char ID;
  AArch64Arm64ECCallLowering() : ModulePass(ID) {
    initializeAArch64Arm64ECCallLoweringPass(*PassRegistry::getPassRegistry());
  }

  Function *buildExitThunk(FunctionType *FnTy, AttributeList Attrs);
  Function *buildEntryThunk(Function *F);
  void lowerCall(CallBase *CB);
  bool processFunction(Function &F,
                       SmallVectorImpl<Function *> &DirectCalledFns);
  bool runOnModule(Module &M) override;

private:
  int cfguard_module_flag = 0;
  FunctionType *GuardFnType = nullptr;
  PointerType *GuardFnPtrType = nullptr;
  Constant *GuardFnCFGlobal = nullptr;
  Constant *GuardFnGlobal = nullptr;
  Module *M = nullptr;

  Type *I8PtrTy;
  Type *I64Ty;
  Type *VoidTy;

  void getThunkType(FunctionType *FT, AttributeList AttrList, bool entry,
                    raw_ostream &Out, FunctionType *&Arm64Ty,
                    FunctionType *&X64Ty);
  void getThunkRetType(FunctionType *FT, AttributeList AttrList,
                       bool EntryThunk, raw_ostream &Out, Type *&Arm64RetTy,
                       Type *&X64RetTy, SmallVectorImpl<Type *> &Arm64ArgTypes,
                       SmallVectorImpl<Type *> &X64ArgTypes);
  void getThunkArgTypes(FunctionType *FT, AttributeList AttrList,
                        bool EntryThunk, raw_ostream &Out,
                        SmallVectorImpl<Type *> &Arm64ArgTypes,
                        SmallVectorImpl<Type *> &X64ArgTypes);
  void canonicalizeThunkType(Type *T, Align Alignment, bool EntryThunk,
                             bool Ret, uint64_t ArgSizeBytes, raw_ostream &Out,
                             Type *&Arm64Ty, Type *&X64Ty);
};

} // end anonymous namespace

void AArch64Arm64ECCallLowering::getThunkType(FunctionType *FT,
                                              AttributeList AttrList,
                                              bool EntryThunk, raw_ostream &Out,
                                              FunctionType *&Arm64Ty,
                                              FunctionType *&X64Ty) {
  Out << (EntryThunk ? "$ientry_thunk$cdecl$" : "$iexit_thunk$cdecl$");

  Type *Arm64RetTy;
  Type *X64RetTy;

  SmallVector<Type *> Arm64ArgTypes;
  SmallVector<Type *> X64ArgTypes;

  // The first argument to a thunk is the called function, stored in x9.
  // For exit thunks, we pass the called function down to the emulator;
  // for entry thunks, we just call the Arm64 function directly.
  if (!EntryThunk)
    Arm64ArgTypes.push_back(I8PtrTy);
  X64ArgTypes.push_back(I8PtrTy);

  getThunkRetType(FT, AttrList, EntryThunk, Out, Arm64RetTy, X64RetTy,
                  Arm64ArgTypes, X64ArgTypes);

  getThunkArgTypes(FT, AttrList, EntryThunk, Out, Arm64ArgTypes, X64ArgTypes);

  Arm64Ty = FunctionType::get(Arm64RetTy, Arm64ArgTypes, false);
  X64Ty = FunctionType::get(X64RetTy, X64ArgTypes, false);
}

void AArch64Arm64ECCallLowering::getThunkArgTypes(
    FunctionType *FT, AttributeList AttrList, bool EntryThunk, raw_ostream &Out,
    SmallVectorImpl<Type *> &Arm64ArgTypes,
    SmallVectorImpl<Type *> &X64ArgTypes) {
  bool HasSretPtr = Arm64ArgTypes.size() > 1;

  Out << "$";
  if (FT->isVarArg()) {
    // We treat the variadic function's thunk as a normal function
    // with the following type on the ARM side:
    //   rettype exitthunk(
    //     ptr x9, ptr x0, i64 x1, i64 x2, i64 x3, ptr x4, i64 x5)
    //
    // that can coverage all types of variadic function.
    // x9 is similar to normal exit thunk, store the called function.
    // x0-x3 is the arguments be stored in registers.
    // x4 is the address of the arguments on the stack.
    // x5 is the size of the arguments on the stack.
    //
    // On the x64 side, it's the same except that x5 isn't set.
    //
    // If both the ARM and X64 sides are sret, there are only three
    // arguments in registers.
    //
    // If the X64 side is sret, but the ARM side isn't, we pass an extra value
    // to/from the X64 side, and let SelectionDAG transform it into a memory
    // location.
    Out << "varargs";

    // x0-x3
    for (int i = HasSretPtr ? 1 : 0; i < 4; i++) {
      Arm64ArgTypes.push_back(I64Ty);
      X64ArgTypes.push_back(I64Ty);
    }

    // x4
    Arm64ArgTypes.push_back(I8PtrTy);
    X64ArgTypes.push_back(I8PtrTy);
    // x5
    Arm64ArgTypes.push_back(I64Ty);
    // FIXME: x5 isn't actually passed/used by the x64 side; revisit once we
    // have proper isel for varargs
    X64ArgTypes.push_back(I64Ty);
    return;
  }

  unsigned I = 0;
  if (HasSretPtr)
    I++;

  if (I == FT->getNumParams()) {
    Out << "v";
    return;
  }

  for (unsigned E = FT->getNumParams(); I != E; ++I) {
    Align ParamAlign = AttrList.getParamAlignment(I).valueOrOne();
#if 0
    // FIXME: Need more information about argument size; see
    // https://reviews.llvm.org/D132926
    uint64_t ArgSizeBytes = AttrList.getParamArm64ECArgSizeBytes(I);
#else
    uint64_t ArgSizeBytes = 0;
#endif
    Type *Arm64Ty, *X64Ty;
    canonicalizeThunkType(FT->getParamType(I), ParamAlign, EntryThunk,
                          /*Ret*/ false, ArgSizeBytes, Out, Arm64Ty, X64Ty);
    Arm64ArgTypes.push_back(Arm64Ty);
    X64ArgTypes.push_back(X64Ty);
  }
}

void AArch64Arm64ECCallLowering::getThunkRetType(
    FunctionType *FT, AttributeList AttrList, bool EntryThunk, raw_ostream &Out,
    Type *&Arm64RetTy, Type *&X64RetTy, SmallVectorImpl<Type *> &Arm64ArgTypes,
    SmallVectorImpl<Type *> &X64ArgTypes) {
  Type *T = FT->getReturnType();
#if 0
  // FIXME: Need more information about argument size; see
  // https://reviews.llvm.org/D132926
  uint64_t ArgSizeBytes = AttrList.getRetArm64ECArgSizeBytes();
#else
  int64_t ArgSizeBytes = 0;
#endif
  if (T->isVoidTy()) {
    if (FT->getNumParams()) {
      auto Attr = AttrList.getParamAttr(0, Attribute::StructRet);
      if (Attr.isValid()) {
        Type *SRetType = Attr.getValueAsType();
        Align SRetAlign = AttrList.getParamAlignment(0).valueOrOne();
        Type *Arm64Ty, *X64Ty;
        canonicalizeThunkType(SRetType, SRetAlign, EntryThunk, /*Ret*/ true,
                              ArgSizeBytes, Out, Arm64Ty, X64Ty);
        Arm64RetTy = VoidTy;
        X64RetTy = VoidTy;
        Arm64ArgTypes.push_back(FT->getParamType(0));
        X64ArgTypes.push_back(FT->getParamType(0));
        return;
      }
    }

    Out << "v";
    Arm64RetTy = VoidTy;
    X64RetTy = VoidTy;
    return;
  }

  canonicalizeThunkType(T, Align(), EntryThunk, /*Ret*/ true, ArgSizeBytes, Out,
                        Arm64RetTy, X64RetTy);
  if (X64RetTy->isPointerTy()) {
    // If the X64 type is canonicalized to a pointer, that means it's
    // passed/returned indirectly. For a return value, that means it's an
    // sret pointer.
    X64ArgTypes.push_back(X64RetTy);
    X64RetTy = VoidTy;
  }
}

void AArch64Arm64ECCallLowering::canonicalizeThunkType(
    Type *T, Align Alignment, bool EntryThunk, bool Ret, uint64_t ArgSizeBytes,
    raw_ostream &Out, Type *&Arm64Ty, Type *&X64Ty) {
  if (T->isFloatTy()) {
    Out << "f";
    Arm64Ty = T;
    X64Ty = T;
    return;
  }

  if (T->isDoubleTy()) {
    Out << "d";
    Arm64Ty = T;
    X64Ty = T;
    return;
  }

  auto &DL = M->getDataLayout();

  if (auto *StructTy = dyn_cast<StructType>(T))
    if (StructTy->getNumElements() == 1)
      T = StructTy->getElementType(0);

  if (T->isArrayTy()) {
    Type *ElementTy = T->getArrayElementType();
    uint64_t ElementCnt = T->getArrayNumElements();
    uint64_t ElementSizePerBytes = DL.getTypeSizeInBits(ElementTy) / 8;
    uint64_t TotalSizeBytes = ElementCnt * ElementSizePerBytes;
    if (ElementTy->isFloatTy() || ElementTy->isDoubleTy()) {
      Out << (ElementTy->isFloatTy() ? "F" : "D") << TotalSizeBytes;
      if (Alignment.value() >= 8 && !T->isPointerTy())
        Out << "a" << Alignment.value();
      Arm64Ty = T;
      if (TotalSizeBytes <= 8) {
        // Arm64 returns small structs of float/double in float registers;
        // X64 uses RAX.
        X64Ty = llvm::Type::getIntNTy(M->getContext(), TotalSizeBytes * 8);
      } else {
        // Struct is passed directly on Arm64, but indirectly on X64.
        X64Ty = Arm64Ty->getPointerTo(0);
      }
      return;
    }
  }

  if ((T->isIntegerTy() || T->isPointerTy()) && DL.getTypeSizeInBits(T) <= 64) {
    Out << "i8";
    Arm64Ty = I64Ty;
    X64Ty = I64Ty;
    return;
  }

  unsigned TypeSize = ArgSizeBytes;
  if (TypeSize == 0)
    TypeSize = DL.getTypeSizeInBits(T) / 8;
  Out << "m";
  if (TypeSize != 4)
    Out << TypeSize;
  if (Alignment.value() >= 8 && !T->isPointerTy())
    Out << "a" << Alignment.value();
  // FIXME: Try to canonicalize Arm64Ty more thoroughly?
  Arm64Ty = T;
  if (TypeSize == 1 || TypeSize == 2 || TypeSize == 4 || TypeSize == 8) {
    // Pass directly in an integer register
    X64Ty = llvm::Type::getIntNTy(M->getContext(), TypeSize * 8);
  } else {
    // Passed directly on Arm64, but indirectly on X64.
    X64Ty = Arm64Ty->getPointerTo(0);
  }
}

Function *AArch64Arm64ECCallLowering::buildExitThunk(FunctionType *FT,
                                                     AttributeList Attrs) {
  SmallString<256> ExitThunkName;
  llvm::raw_svector_ostream ExitThunkStream(ExitThunkName);
  FunctionType *Arm64Ty, *X64Ty;
  getThunkType(FT, Attrs, /*EntryThunk*/ false, ExitThunkStream, Arm64Ty,
               X64Ty);
  if (Function *F = M->getFunction(ExitThunkName))
    return F;

  Function *F = Function::Create(Arm64Ty, GlobalValue::LinkOnceODRLinkage, 0,
                                 ExitThunkName, M);
  F->setCallingConv(CallingConv::ARM64EC_Thunk_Native);
  F->setSection(".wowthk$aa");
  F->setComdat(M->getOrInsertComdat(ExitThunkName));
  // Copy MSVC, and always set up a frame pointer. (Maybe this isn't necessary.)
  F->addFnAttr("frame-pointer", "all");
  // Only copy sret from the first argument. For C++ instance methods, clang can
  // stick an sret marking on a later argument, but it doesn't actually affect
  // the ABI, so we can omit it. This avoids triggering a verifier assertion.
  if (FT->getNumParams()) {
    auto SRet = Attrs.getParamAttr(0, Attribute::StructRet);
    if (SRet.isValid())
      F->addParamAttr(1, SRet);
  }
  // FIXME: Copy anything other than sret?  Shouldn't be necessary for normal
  // C ABI, but might show up in other cases.
  BasicBlock *BB = BasicBlock::Create(M->getContext(), "", F);
  IRBuilder<> IRB(BB);
  PointerType *DispatchPtrTy =
      FunctionType::get(IRB.getVoidTy(), false)->getPointerTo(0);
  Value *CalleePtr = M->getOrInsertGlobal(
      "__os_arm64x_dispatch_call_no_redirect", DispatchPtrTy);
  Value *Callee = IRB.CreateLoad(DispatchPtrTy, CalleePtr);
  auto &DL = M->getDataLayout();
  SmallVector<Value *> Args;

  // Pass the called function in x9.
  Args.push_back(F->arg_begin());

  Type *RetTy = Arm64Ty->getReturnType();
  if (RetTy != X64Ty->getReturnType()) {
    // If the return type is an array or struct, translate it. Values of size
    // 8 or less go into RAX; bigger values go into memory, and we pass a
    // pointer.
    if (DL.getTypeStoreSize(RetTy) > 8) {
      Args.push_back(IRB.CreateAlloca(RetTy));
    }
  }

  for (auto &Arg : make_range(F->arg_begin() + 1, F->arg_end())) {
    // Translate arguments from AArch64 calling convention to x86 calling
    // convention.
    //
    // For simple types, we don't need to do any translation: they're
    // represented the same way. (Implicit sign extension is not part of
    // either convention.)
    //
    // The big thing we have to worry about is struct types... but
    // fortunately AArch64 clang is pretty friendly here: the cases that need
    // translation are always passed as a struct or array. (If we run into
    // some cases where this doesn't work, we can teach clang to mark it up
    // with an attribute.)
    //
    // The first argument is the called function, stored in x9.
    if (Arg.getType()->isArrayTy() || Arg.getType()->isStructTy() ||
        DL.getTypeStoreSize(Arg.getType()) > 8) {
      Value *Mem = IRB.CreateAlloca(Arg.getType());
      IRB.CreateStore(&Arg, Mem);
      if (DL.getTypeStoreSize(Arg.getType()) <= 8) {
        Type *IntTy = IRB.getIntNTy(DL.getTypeStoreSizeInBits(Arg.getType()));
        Args.push_back(IRB.CreateLoad(
            IntTy, IRB.CreateBitCast(Mem, IntTy->getPointerTo(0))));
      } else
        Args.push_back(Mem);
    } else {
      Args.push_back(&Arg);
    }
  }
  // FIXME: Transfer necessary attributes? sret? anything else?

  Callee = IRB.CreateBitCast(Callee, X64Ty->getPointerTo(0));
  CallInst *Call = IRB.CreateCall(X64Ty, Callee, Args);
  Call->setCallingConv(CallingConv::ARM64EC_Thunk_X64);

  Value *RetVal = Call;
  if (RetTy != X64Ty->getReturnType()) {
    // If we rewrote the return type earlier, convert the return value to
    // the proper type.
    if (DL.getTypeStoreSize(RetTy) > 8) {
      RetVal = IRB.CreateLoad(RetTy, Args[1]);
    } else {
      Value *CastAlloca = IRB.CreateAlloca(RetTy);
      IRB.CreateStore(Call, IRB.CreateBitCast(
                                CastAlloca, Call->getType()->getPointerTo(0)));
      RetVal = IRB.CreateLoad(RetTy, CastAlloca);
    }
  }

  if (RetTy->isVoidTy())
    IRB.CreateRetVoid();
  else
    IRB.CreateRet(RetVal);
  return F;
}

Function *AArch64Arm64ECCallLowering::buildEntryThunk(Function *F) {
  SmallString<256> EntryThunkName;
  llvm::raw_svector_ostream EntryThunkStream(EntryThunkName);
  FunctionType *Arm64Ty, *X64Ty;
  getThunkType(F->getFunctionType(), F->getAttributes(), /*EntryThunk*/ true,
               EntryThunkStream, Arm64Ty, X64Ty);
  if (Function *F = M->getFunction(EntryThunkName))
    return F;

  Function *Thunk = Function::Create(X64Ty, GlobalValue::LinkOnceODRLinkage, 0,
                                     EntryThunkName, M);
  Thunk->setCallingConv(CallingConv::ARM64EC_Thunk_X64);
  Thunk->setSection(".wowthk$aa");
  Thunk->setComdat(M->getOrInsertComdat(EntryThunkName));
  // Copy MSVC, and always set up a frame pointer. (Maybe this isn't necessary.)
  Thunk->addFnAttr("frame-pointer", "all");

  auto &DL = M->getDataLayout();
  BasicBlock *BB = BasicBlock::Create(M->getContext(), "", Thunk);
  IRBuilder<> IRB(BB);

  Type *RetTy = Arm64Ty->getReturnType();
  Type *X64RetType = X64Ty->getReturnType();

  bool TransformDirectToSRet = X64RetType->isVoidTy() && !RetTy->isVoidTy();
  unsigned ThunkArgOffset = TransformDirectToSRet ? 2 : 1;

  // Translate arguments to call.
  SmallVector<Value *> Args;
  for (unsigned i = ThunkArgOffset, e = Thunk->arg_size(); i != e; ++i) {
    Value *Arg = Thunk->getArg(i);
    Type *ArgTy = Arm64Ty->getParamType(i - ThunkArgOffset);
    if (ArgTy->isArrayTy() || ArgTy->isStructTy() ||
        DL.getTypeStoreSize(ArgTy) > 8) {
      // Translate array/struct arguments to the expected type.
      if (DL.getTypeStoreSize(ArgTy) <= 8) {
        Value *CastAlloca = IRB.CreateAlloca(ArgTy);
        IRB.CreateStore(Arg, IRB.CreateBitCast(
                                 CastAlloca, Arg->getType()->getPointerTo(0)));
        Arg = IRB.CreateLoad(ArgTy, CastAlloca);
      } else {
        Arg = IRB.CreateLoad(ArgTy,
                             IRB.CreateBitCast(Arg, ArgTy->getPointerTo(0)));
      }
    }
    Args.push_back(Arg);
  }

  // Call the function passed to the thunk.
  Value *Callee = Thunk->getArg(0);
  Callee = IRB.CreateBitCast(Callee, Arm64Ty->getPointerTo(0));
  Value *Call = IRB.CreateCall(Arm64Ty, Callee, Args);

  Value *RetVal = Call;
  if (TransformDirectToSRet) {
    IRB.CreateStore(RetVal,
                    IRB.CreateBitCast(Thunk->getArg(1),
                                      RetVal->getType()->getPointerTo(0)));
  } else if (X64RetType != RetTy) {
    Value *CastAlloca = IRB.CreateAlloca(X64RetType);
    IRB.CreateStore(
        Call, IRB.CreateBitCast(CastAlloca, Call->getType()->getPointerTo(0)));
    RetVal = IRB.CreateLoad(X64RetType, CastAlloca);
  }

  // Return to the caller.  Note that the isel has code to translate this
  // "ret" to a tail call to __os_arm64x_dispatch_ret.  (Alternatively, we
  // could emit a tail call here, but that would require a dedicated calling
  // convention, which seems more complicated overall.)
  if (X64RetType->isVoidTy())
    IRB.CreateRetVoid();
  else
    IRB.CreateRet(RetVal);

  return Thunk;
}

void AArch64Arm64ECCallLowering::lowerCall(CallBase *CB) {
  assert(Triple(CB->getModule()->getTargetTriple()).isOSWindows() &&
         "Only applicable for Windows targets");

  IRBuilder<> B(CB);
  Value *CalledOperand = CB->getCalledOperand();

  // If the indirect call is called within catchpad or cleanuppad,
  // we need to copy "funclet" bundle of the call.
  SmallVector<llvm::OperandBundleDef, 1> Bundles;
  if (auto Bundle = CB->getOperandBundle(LLVMContext::OB_funclet))
    Bundles.push_back(OperandBundleDef(*Bundle));

  // Load the global symbol as a pointer to the check function.
  Value *GuardFn;
  if (cfguard_module_flag == 2 && !CB->hasFnAttr("guard_nocf"))
    GuardFn = GuardFnCFGlobal;
  else
    GuardFn = GuardFnGlobal;
  LoadInst *GuardCheckLoad = B.CreateLoad(GuardFnPtrType, GuardFn);

  // Create new call instruction. The CFGuard check should always be a call,
  // even if the original CallBase is an Invoke or CallBr instruction.
  Function *Thunk = buildExitThunk(CB->getFunctionType(), CB->getAttributes());
  CallInst *GuardCheck =
      B.CreateCall(GuardFnType, GuardCheckLoad,
                   {B.CreateBitCast(CalledOperand, B.getInt8PtrTy()),
                    B.CreateBitCast(Thunk, B.getInt8PtrTy())},
                   Bundles);

  // Ensure that the first argument is passed in the correct register
  // (e.g. ECX on 32-bit X86 targets).
  GuardCheck->setCallingConv(CallingConv::CFGuard_Check);

  Value *GuardRetVal = B.CreateBitCast(GuardCheck, CalledOperand->getType());
  CB->setCalledOperand(GuardRetVal);
}

bool AArch64Arm64ECCallLowering::runOnModule(Module &Mod) {
  if (!GenerateThunks)
    return false;

  M = &Mod;

  // Check if this module has the cfguard flag and read its value.
  if (auto *MD =
          mdconst::extract_or_null<ConstantInt>(M->getModuleFlag("cfguard")))
    cfguard_module_flag = MD->getZExtValue();

  I8PtrTy = Type::getInt8PtrTy(M->getContext());
  I64Ty = Type::getInt64Ty(M->getContext());
  VoidTy = Type::getVoidTy(M->getContext());

  GuardFnType = FunctionType::get(I8PtrTy, {I8PtrTy, I8PtrTy}, false);
  GuardFnPtrType = PointerType::get(GuardFnType, 0);
  GuardFnCFGlobal =
      M->getOrInsertGlobal("__os_arm64x_check_icall_cfg", GuardFnPtrType);
  GuardFnGlobal =
      M->getOrInsertGlobal("__os_arm64x_check_icall", GuardFnPtrType);

  SmallVector<Function *> DirectCalledFns;
  for (Function &F : Mod)
    if (!F.isDeclaration() &&
        F.getCallingConv() != CallingConv::ARM64EC_Thunk_Native &&
        F.getCallingConv() != CallingConv::ARM64EC_Thunk_X64)
      processFunction(F, DirectCalledFns);

  struct ThunkInfo {
    Constant *Src;
    Constant *Dst;
    unsigned Kind;
  };
  SmallVector<ThunkInfo> ThunkMapping;
  for (Function &F : Mod) {
    if (!F.isDeclaration() && !F.hasLocalLinkage() &&
        F.getCallingConv() != CallingConv::ARM64EC_Thunk_Native &&
        F.getCallingConv() != CallingConv::ARM64EC_Thunk_X64) {
      if (!F.hasComdat())
        F.setComdat(Mod.getOrInsertComdat(F.getName()));
      ThunkMapping.push_back({&F, buildEntryThunk(&F), 1});
    }
  }
  for (Function *F : DirectCalledFns) {
    ThunkMapping.push_back(
        {F, buildExitThunk(F->getFunctionType(), F->getAttributes()), 4});
  }

  if (!ThunkMapping.empty()) {
    Type *VoidPtr = Type::getInt8PtrTy(M->getContext());
    SmallVector<Constant *> ThunkMappingArrayElems;
    for (ThunkInfo &Thunk : ThunkMapping) {
      ThunkMappingArrayElems.push_back(ConstantStruct::getAnon(
          {ConstantExpr::getBitCast(Thunk.Src, VoidPtr),
           ConstantExpr::getBitCast(Thunk.Dst, VoidPtr),
           ConstantInt::get(M->getContext(), APInt(32, Thunk.Kind))}));
    }
    Constant *ThunkMappingArray = ConstantArray::get(
        llvm::ArrayType::get(ThunkMappingArrayElems[0]->getType(),
                             ThunkMappingArrayElems.size()),
        ThunkMappingArrayElems);
    new GlobalVariable(Mod, ThunkMappingArray->getType(), /*isConstant*/ false,
                       GlobalValue::ExternalLinkage, ThunkMappingArray,
                       "llvm.arm64ec.symbolmap");
  }

  return true;
}

bool AArch64Arm64ECCallLowering::processFunction(
    Function &F, SmallVectorImpl<Function *> &DirectCalledFns) {
  SmallVector<CallBase *, 8> IndirectCalls;

  // For ARM64EC targets, a function definition's name is mangled differently
  // from the normal symbol. We currently have no representation of this sort
  // of symbol in IR, so we change the name to the mangled name, then store
  // the unmangled name as metadata.  Later passes that need the unmangled
  // name (emitting the definition) can grab it from the metadata.
  //
  // FIXME: Handle functions with weak linkage?
  if (F.hasExternalLinkage() || F.hasWeakLinkage() || F.hasLinkOnceLinkage()) {
    if (std::optional<std::string> MangledName =
            getArm64ECMangledFunctionName(F.getName().str())) {
      F.setMetadata("arm64ec_unmangled_name", MDNode::get(M->getContext(), MDString::get(M->getContext(), F.getName())));
      if (F.hasComdat() && F.getComdat()->getName() == F.getName()) {
        Comdat *MangledComdat = M->getOrInsertComdat(MangledName.value());
        SmallVector<GlobalObject*> ComdatUsers = to_vector(F.getComdat()->getUsers());
        for (GlobalObject *User : ComdatUsers)
          User->setComdat(MangledComdat);
      }
      F.setName(MangledName.value());
    }
  }

  // Iterate over the instructions to find all indirect call/invoke/callbr
  // instructions. Make a separate list of pointers to indirect
  // call/invoke/callbr instructions because the original instructions will be
  // deleted as the checks are added.
  for (BasicBlock &BB : F) {
    for (Instruction &I : BB) {
      auto *CB = dyn_cast<CallBase>(&I);
      if (!CB || CB->getCallingConv() == CallingConv::ARM64EC_Thunk_X64 ||
          CB->isInlineAsm())
        continue;

      // We need to instrument any call that isn't directly calling an
      // ARM64 function.
      //
      // FIXME: isDSOLocal() doesn't do what we want; even if the symbol is
      // technically local, automatic dllimport means the function it refers
      // to might not be.
      //
      // FIXME: If a function is dllimport, we can just mark up the symbol
      // using hybmp$x, and everything just works.  If the function is not
      // marked dllimport, we can still mark up the symbol, but we somehow
      // need an extra stub to compute the correct callee. Not really
      // understanding how this works.
      //
      // FIXME: getCalledFunction() fails if there's a bitcast (e.g.
      // unprototyped functions in C)
      if (Function *F = CB->getCalledFunction()) {
        if (!LowerDirectToIndirect || F->hasLocalLinkage() || F->isIntrinsic() ||
            !F->isDeclaration())
          continue;

        DirectCalledFns.push_back(F);
        continue;
      }

      IndirectCalls.push_back(CB);
      ++Arm64ECCallsLowered;
    }
  }

  if (IndirectCalls.empty())
    return false;

  for (CallBase *CB : IndirectCalls)
    lowerCall(CB);

  return true;
}

char AArch64Arm64ECCallLowering::ID = 0;
INITIALIZE_PASS(AArch64Arm64ECCallLowering, "Arm64ECCallLowering",
                "AArch64Arm64ECCallLowering", false, false)

ModulePass *llvm::createAArch64Arm64ECCallLoweringPass() {
  return new AArch64Arm64ECCallLowering;
}
