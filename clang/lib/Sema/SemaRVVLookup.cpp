//===-- SemaRVVLookup.cpp - Name Lookup for RISC-V Vector Intrinsic -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file implements name lookup for RISC-V vector intrinsic.
//
//===----------------------------------------------------------------------===//

#include "clang/AST/ASTContext.h"
#include "clang/AST/CXXInheritance.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclLookups.h"
#include "clang/AST/DeclObjC.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/Expr.h"
#include "clang/AST/ExprCXX.h"
#include "clang/Basic/Builtins.h"
#include "clang/Basic/FileManager.h"
#include "clang/Basic/LangOptions.h"
#include "clang/Basic/TargetBuiltins.h"
#include "clang/Basic/TargetInfo.h"
#include "clang/Lex/HeaderSearch.h"
#include "clang/Lex/ModuleLoader.h"
#include "clang/Lex/Preprocessor.h"
#include "clang/Sema/DeclSpec.h"
#include "clang/Sema/Lookup.h"
#include "clang/Sema/Overload.h"
#include "clang/Sema/Scope.h"
#include "clang/Sema/ScopeInfo.h"
#include "clang/Sema/Sema.h"
#include "clang/Sema/SemaInternal.h"
#include "clang/Sema/TemplateDeduction.h"
#include "clang/Sema/TypoCorrection.h"
#include "clang/Support/RISCVVIntrinsicUtils.h"
#include "llvm/ADT/BitmaskEnum.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/TinyPtrVector.h"
#include "llvm/ADT/edit_distance.h"
#include "llvm/Support/ErrorHandling.h"
#include <algorithm>
#include <iterator>
#include <list>
#include <set>
#include <utility>
#include <vector>

using namespace llvm;
using namespace clang;
using namespace clang::RISCV;

namespace {

// RVVRequire should be sync with target features, but only
// required features used in riscv_vector.td.
enum class RVVRequire : uint8_t {
  None = 0,
  RV64 = 1 << 0,
  FullMultiply = 1 << 1,

  LLVM_MARK_AS_BITMASK_ENUM(FullMultiply)
};

// Raw RVV intrinsic info, used to expand later.
// This struct is highly compact for minimized code size, and should sync
// with RISCVVEmitter.cpp.
struct RVVIntrinsicRecord {
  // Intrinsic name, e.g. vadd_vv
  const char *Name;

  // Overloaded intrinsic name, could be empty if can be computed from Name
  // e.g. vadd
  const char *OverloadedName;

  // Prototype for this intrinsic, index of RVVSignatureTable.
  uint16_t PrototypeIndex;

  // Prototype for masked intrinsic, index of RVVSignatureTable.
  uint16_t MaskedPrototypeIndex;

  // Suffix of intrinsic name, index of RVVSignatureTable.
  uint16_t SuffixIndex;

  // Suffix of overloaded intrinsic name, index of RVVSignatureTable.
  uint16_t OverloadedSuffixIndex;

  // Length of the prototype.
  uint8_t PrototypeLength;

  // Length of prototype of masked intrinsic.
  uint8_t MaskedPrototypeLength;

  // Length of intrinsic name suffix.
  uint8_t SuffixLength;

  // Length of overloaded intrinsic suffix.
  uint8_t OverloadedSuffixSize;

  // Required target features for this intrinsic.
  RVVRequire RequiredExtension;

  // Supported type, mask of BasicType
  uint8_t TypeRangeMask;

  // Supported LMUL.
  uint8_t Log2LMULMask;

  // Number of field, large than 1 if it's segment load/store.
  uint8_t NF;
};

// Function definition of a RVV intrinsic
struct RVVIntrinsicDef {
  /// Full function name with suffix, e.g. vadd_vv_i32m1.
  std::string Name;

  /// Overloaded function name, e.g. vadd.
  std::string OverloadName;

  /// Mapping to which clang built-in function, e.g. __builtin_rvv_vadd
  std::string BuiltinName;

  /// Function signature, first element is return type.
  RVVTypes Signature;
};

struct RVVOverloadIntrinsicDef {
  // Index of RVVIntrinsicManager::IntrinsicList.
  SmallVector<size_t, 8> Indexs;
};

LLVM_ENABLE_BITMASK_ENUMS_IN_NAMESPACE();
} // namespace

static const TypeProfile RVVSignatureTable[] = {
#define DECL_SIGNATURE_TABLE
#include "clang/Basic/riscv_vector_builtin_sema.inc"
#undef DECL_SIGNATURE_TABLE
};

static const RVVIntrinsicRecord RVVIntrinsicRecords[] = {
#define DECL_INTRINSIC_RECORDS
#include "clang/Basic/riscv_vector_builtin_sema.inc"
#undef DECL_INTRINSIC_RECORDS
};

static ArrayRef<TypeProfile> ProtoSeq2ArrayRef(uint16_t Index, uint8_t Length) {
  return ArrayRef<TypeProfile>(&RVVSignatureTable[Index], Length);
}

static QualType RVVType2Qual(ASTContext &Context, const RVVType *Type) {
  QualType QT;
  switch (Type->getScalarType()) {
  case ScalarTypeKind::Void:
    QT = Context.VoidTy;
    break;
  case ScalarTypeKind::Size_t:
    QT = Context.getSizeType();
    break;
  case ScalarTypeKind::Ptrdiff_t:
    QT = Context.getPointerDiffType();
    break;
  case ScalarTypeKind::UnsignedLong:
    QT = Context.UnsignedLongTy;
    break;
  case ScalarTypeKind::SignedLong:
    QT = Context.LongTy;
    break;
  case ScalarTypeKind::Boolean:
    QT = Context.BoolTy;
    break;
  case ScalarTypeKind::SignedInteger:
    QT = Context.getIntTypeForBitwidth(Type->getElementBitwidth(), true);
    break;
  case ScalarTypeKind::UnsignedInteger:
    QT = Context.getIntTypeForBitwidth(Type->getElementBitwidth(), false);
    break;
  case ScalarTypeKind::Float:
    switch (Type->getElementBitwidth()) {
    case 64:
      QT = Context.DoubleTy;
      break;
    case 32:
      QT = Context.FloatTy;
      break;
    case 16:
      QT = Context.Float16Ty;
      break;
    }
    break;
  default:
    return QT;
  }
  if (Type->isVector())
    QT = Context.getScalableVectorType(QT, Type->getScale().getValue());

  if (Type->isConstant()) {
    QT = Context.getConstType(QT);
  }

  // Transform the type to a pointer as the last step, if necessary.
  if (Type->isPointer()) {
    QT = Context.getPointerType(QT);
  }

  return QT;
}

static void
CreateRVVIntrinsicDecl(Sema &S, LookupResult &LR, IdentifierInfo *II,
                       Preprocessor &PP, unsigned Index,
                       std::vector<RVVIntrinsicDef> &RVVIntrinsicList,
                       bool IsOverload) {
  ASTContext &Context = S.Context;
  RVVIntrinsicDef &IDef = RVVIntrinsicList[Index];
  auto Sigs = IDef.Signature;
  size_t SigLength = Sigs.size();
  auto ReturnType = Sigs[0];
  QualType RetType = RVVType2Qual(Context, ReturnType);
  SmallVector<QualType, 8> ArgTypes;
  QualType BuiltinFuncType;

  // Skip return type, and convert RVVType to QualType for arguments.
  for (size_t i = 1; i < SigLength; ++i) {
    ArgTypes.push_back(RVVType2Qual(Context, Sigs[i]));
  }

  FunctionProtoType::ExtProtoInfo PI(
      Context.getDefaultCallingConvention(false, false, true));

  PI.Variadic = false;

  SourceLocation Loc = LR.getNameLoc();
  BuiltinFuncType = Context.getFunctionType(RetType, ArgTypes, PI);
  DeclContext *Parent = Context.getTranslationUnitDecl();

  FunctionDecl *RVVIntrinsicDecl = FunctionDecl::Create(
      Context, Parent, Loc, Loc, II, BuiltinFuncType, /*TInfo=*/nullptr,
      SC_Extern, S.getCurFPFeatures().isFPConstrained(), false,
      BuiltinFuncType->isFunctionProtoType());

  // Create Decl objects for each parameter, adding them to the
  // FunctionDecl.
  const auto *FP = cast<FunctionProtoType>(BuiltinFuncType);
  SmallVector<ParmVarDecl *, 8> ParmList;
  for (unsigned IParm = 0, e = FP->getNumParams(); IParm != e; ++IParm) {
    ParmVarDecl *Parm =
        ParmVarDecl::Create(Context, RVVIntrinsicDecl, Loc, Loc, nullptr,
                            FP->getParamType(IParm), nullptr, SC_None, nullptr);
    Parm->setScopeInfo(0, IParm);
    ParmList.push_back(Parm);
  }
  RVVIntrinsicDecl->setParams(ParmList);

  // Add function attributes.
  if (IsOverload)
    RVVIntrinsicDecl->addAttr(OverloadableAttr::CreateImplicit(Context));

  // Setup alias to __builtin_rvv_*
  auto &IntrinsicII = PP.getIdentifierTable().get(IDef.BuiltinName);
  RVVIntrinsicDecl->addAttr(
      BuiltinAliasAttr::CreateImplicit(S.Context, &IntrinsicII));

  // Add to symbol table.
  LR.addDecl(RVVIntrinsicDecl);
}

namespace {
class RVVIntrinsicManager {
private:
  // List of all RVV intrinsic.
  std::vector<RVVIntrinsicDef> IntrinsicList;
  // Mapping function name to index of IntrinsicList.
  StringMap<size_t> Intrinsics;
  // Mapping function name to RVVOverloadIntrinsicDef.
  StringMap<RVVOverloadIntrinsicDef> OverloadIntrinsics;

  ASTContext &Context;

  // Create IntrinsicList
  void InitIntrinsicList();

  // Create RVVIntrinsicDef.
  void InitRVVIntrinsic(const RVVIntrinsicRecord &Record, StringRef SuffixStr,
                        StringRef OverloadedSuffixStr, bool IsMask,
                        RVVTypes &Types);

public:
  RVVIntrinsicManager(ASTContext &Context) : Context(Context) {
    InitIntrinsicList();
  }

  bool CreateIntrinsicIfFound(Sema &S, LookupResult &LR, IdentifierInfo *II,
                              Preprocessor &PP);
};
} // namespace

void RVVIntrinsicManager::InitIntrinsicList() {
  const TargetInfo &TI = Context.getTargetInfo();
  bool HasVectorFloat32 = TI.hasFeature("zve32f");
  bool HasVectorFloat64 = TI.hasFeature("zve64d");
  bool HasZvfh = TI.hasFeature("experimental-zvfh");
  bool HasRV64 = TI.hasFeature("64bit");
  bool HasFullMultiply = TI.hasFeature("v");

  for (auto &Record : RVVIntrinsicRecords) {
    // Create Intrinsics for each type and LMUL.
    BasicType BaseType = BasicType::Unknown;
    auto ProtoSeq =
        ProtoSeq2ArrayRef(Record.PrototypeIndex, Record.PrototypeLength);
    auto ProtoMaskSeq = ProtoSeq2ArrayRef(Record.MaskedPrototypeIndex,
                                          Record.MaskedPrototypeLength);
    auto SuffixProto =
        ProtoSeq2ArrayRef(Record.SuffixIndex, Record.SuffixLength);
    auto MangledSuffixProto = ProtoSeq2ArrayRef(Record.OverloadedSuffixIndex,
                                                Record.OverloadedSuffixSize);
    for (int TypeRangeMaskShift = 0;
         TypeRangeMaskShift <= static_cast<int>(BasicType::MaxOffset);
         ++TypeRangeMaskShift) {
      int BaseTypeI = 1 << TypeRangeMaskShift;
      BaseType = static_cast<BasicType>(BaseTypeI);

      if ((BaseTypeI & Record.TypeRangeMask) != BaseTypeI)
        continue;

      // Check requirement.
      if (BaseType == BasicType::Float16 && !HasZvfh)
        continue;

      if (BaseType == BasicType::Float32 && !HasVectorFloat32)
        continue;

      if (BaseType == BasicType::Float64 && !HasVectorFloat64)
        continue;

      if (((Record.RequiredExtension & RVVRequire::RV64) == RVVRequire::RV64) &&
          !HasRV64)
        continue;

      if ((BaseType == BasicType::Int64) &&
          ((Record.RequiredExtension & RVVRequire::FullMultiply) ==
           RVVRequire::FullMultiply) &&
          !HasFullMultiply)
        continue;

      for (int Log2LMUL = -3; Log2LMUL <= 3; Log2LMUL++) {
        if (!(Record.Log2LMULMask & (1 << (Log2LMUL + 3)))) {
          continue;
        }
        Optional<RVVTypes> Types =
            RVVType::computeTypes(BaseType, Log2LMUL, Record.NF, ProtoSeq);

        // Ignored to create new intrinsic if there are any illegal types.
        if (!Types.hasValue()) {
          continue;
        }

        auto SuffixStr =
            RVVIntrinsic::getSuffixStr(BaseType, Log2LMUL, SuffixProto);
        auto OverloadedSuffixStr =
            RVVIntrinsic::getSuffixStr(BaseType, Log2LMUL, MangledSuffixProto);

        // Create non-masked intrinsic.
        InitRVVIntrinsic(Record, SuffixStr, OverloadedSuffixStr, false, *Types);

        bool HasMask = Record.MaskedPrototypeLength != 0;

        if (HasMask) {
          // Create masked intrinsic.
          Optional<RVVTypes> MaskTypes = RVVType::computeTypes(
              BaseType, Log2LMUL, Record.NF, ProtoMaskSeq);

          InitRVVIntrinsic(Record, SuffixStr, OverloadedSuffixStr, true,
                           *MaskTypes);
        }
      }
    }
  }
}

// Compute name and signatures for intrinsic with practical types.
void RVVIntrinsicManager::InitRVVIntrinsic(const RVVIntrinsicRecord &Record,
                                           StringRef SuffixStr,
                                           StringRef OverloadedSuffixStr,
                                           bool IsMask, RVVTypes &Signature) {
  // Function name, e.g. vadd_vv_i32m1.
  std::string Name = Record.Name;
  if (!SuffixStr.empty())
    Name += "_" + SuffixStr.str();

  if (IsMask) {
    Name += "_m";
  }

  // Overloaded function name, e.g. vadd.
  std::string OverloadedName;
  if (!Record.OverloadedName)
    OverloadedName = StringRef(Record.Name).split("_").first.str();
  else
    OverloadedName = Record.OverloadedName;
  if (!OverloadedSuffixStr.empty())
    OverloadedName += "_" + OverloadedSuffixStr.str();

  // clang built-in function name, e.g. __builtin_rvv_vadd.
  std::string BuiltinName = "__builtin_rvv_" + std::string(Record.Name);
  if (IsMask) {
    BuiltinName += "_m";
  }

  // Put into IntrinsicList.
  size_t Index = IntrinsicList.size();
  IntrinsicList.push_back({Name, OverloadedName, BuiltinName, Signature});

  // Creating mapping to Intrinsics.
  Intrinsics.insert({Name, Index});

  // Get the RVVOverloadIntrinsicDef.
  RVVOverloadIntrinsicDef &OverloadIntrinsicDef =
      OverloadIntrinsics[OverloadedName];

  // And added the index.
  OverloadIntrinsicDef.Indexs.push_back(Index);
}

bool RVVIntrinsicManager::CreateIntrinsicIfFound(Sema &S, LookupResult &LR,
                                                 IdentifierInfo *II,
                                                 Preprocessor &PP) {
  StringRef Name = II->getName();

  // Lookup the function name from the overload intrinsics first.
  auto OvIItr = OverloadIntrinsics.find(Name);
  if (OvIItr != OverloadIntrinsics.end()) {
    auto OvIntrinsicDef = OvIItr->second;
    for (auto Index : OvIntrinsicDef.Indexs) {
      CreateRVVIntrinsicDecl(S, LR, II, PP, Index, IntrinsicList,
                             /*IsOverload*/ true);
    }

    // If we added overloads, need to resolve the lookup result.
    LR.resolveKind();
    return true;
  }

  // Lookup the function name from the intrinsics.
  auto Itr = Intrinsics.find(Name);
  if (Itr != Intrinsics.end()) {
    CreateRVVIntrinsicDecl(S, LR, II, PP, Itr->second, IntrinsicList,
                           /*IsOverload*/ false);
    return true;
  }

  // It's not RVV intrinsics.
  return false;
}

namespace clang {
bool GetRVVBuiltinInfo(Sema &S, LookupResult &LR, IdentifierInfo *II,
                       Preprocessor &PP) {
  static std::unique_ptr<RVVIntrinsicManager> IntrinsicManager =
      std::make_unique<RVVIntrinsicManager>(S.Context);

  return IntrinsicManager->CreateIntrinsicIfFound(S, LR, II, PP);
}
} // namespace clang
