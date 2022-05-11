//===- RISCVVEmitter.cpp - Generate riscv_vector.h for use with clang -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This tablegen backend is responsible for emitting riscv_vector.h which
// includes a declaration and definition of each intrinsic functions specified
// in https://github.com/riscv/rvv-intrinsic-doc.
//
// See also the documentation in include/clang/Basic/riscv_vector.td.
//
//===----------------------------------------------------------------------===//

#include "clang/Support/RISCVVIntrinsicUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/TableGen/Error.h"
#include "llvm/TableGen/Record.h"
#include <numeric>

using namespace llvm;
using namespace clang::RISCV;

namespace {
struct SemaRecord {
  // Intrinsic name, e.g. vadd_vv
  std::string Name;

  // Overloaded intrinsic name, could be empty if can be computed from Name
  // e.g. vadd
  std::string OverloadedName;

  // Supported type, mask of BasicType
  unsigned TypeRangeMask;

  // Supported LMUL.
  unsigned Log2LMULMask;

  // Required target features for this intrinsic.
  std::vector<std::string> RequiredFeatures;

  // Prototype for this intrinsic.
  SmallVector<TypeProfile> Prototype;

  // Prototype for masked intrinsic.
  SmallVector<TypeProfile> MaskedPrototype;

  // Suffix of intrinsic name.
  SmallVector<TypeProfile> Suffix;

  // Suffix of overloaded intrinsic name.
  SmallVector<TypeProfile> OverloadedSuffix;

  // Number of field, large than 1 if it's segment load/store.
  unsigned NF;
};

class RVVEmitter {
private:
  RecordKeeper &Records;
  // Concat BasicType, LMUL and Proto as key
  StringMap<RVVType> LegalTypes;
  StringSet<> IllegalTypes;

  std::vector<SemaRecord> SemaRecords;
  std::vector<TypeProfile> SemaSignatureTable;

public:
  RVVEmitter(RecordKeeper &R) : Records(R) {}

  /// Emit riscv_vector.h
  void createHeader(raw_ostream &o);

  /// Emit all the __builtin prototypes and code needed by Sema.
  void createBuiltins(raw_ostream &o);

  /// Emit all the information needed to map builtin -> LLVM IR intrinsic.
  void createCodeGen(raw_ostream &o);

  /// Emit all the information needed by SemaRVVLookup.cpp.
  /// We've large number of intrinsic function for RVV, creating a customized
  /// could speed up the compilation time.
  void createSema(raw_ostream &o);

private:
  /// Create all intrinsics and add them to \p Out
  void createRVVIntrinsics(std::vector<std::unique_ptr<RVVIntrinsic>> &Out);
  /// Print HeaderCode in RVVHeader Record to \p Out
  void printHeaderCode(raw_ostream &OS);

  /// Construct a compressed signature table used for createSema.
  void ConstructSemaSignatureTable();
  /// Get index from the signature table.
  unsigned GetSemaSignatureIndex(const SmallVector<TypeProfile> &Signature);

  /// Emit all SemaRecord for SemaRVVLookup.cpp.
  void EmitSemaRecords(raw_ostream &OS);

  /// Emit signature table for SemaRVVLookup.cpp.
  void EmitSemaSignatureTable(raw_ostream &OS);
};

} // namespace

static BasicType ParseBasicType(char c) {
  switch (c) {
  case 'c':
    return BasicType::Int8;
    break;
  case 's':
    return BasicType::Int16;
    break;
  case 'i':
    return BasicType::Int32;
    break;
  case 'l':
    return BasicType::Int64;
    break;
  case 'x':
    return BasicType::Float16;
    break;
  case 'f':
    return BasicType::Float32;
    break;
  case 'd':
    return BasicType::Float64;
    break;

  default:
    return BasicType::Unknown;
  }
}

void emitCodeGenSwitchBody(const RVVIntrinsic *RVVI, raw_ostream &OS) {
  if (!RVVI->getIRName().empty())
    OS << "  ID = Intrinsic::riscv_" + RVVI->getIRName() + ";\n";
  if (RVVI->getNF() >= 2)
    OS << "  NF = " + utostr(RVVI->getNF()) + ";\n";
  if (RVVI->hasManualCodegen()) {
    OS << RVVI->getManualCodegen();
    OS << "break;\n";
    return;
  }

  if (RVVI->isMasked()) {
    if (RVVI->hasVL()) {
      OS << "  std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end() - 1);\n";
      if (RVVI->hasPolicyOperand())
        OS << "  Ops.push_back(ConstantInt::get(Ops.back()->getType(),"
              " TAIL_UNDISTURBED));\n";
    } else {
      OS << "  std::rotate(Ops.begin(), Ops.begin() + 1, Ops.end());\n";
    }
  } else {
    if (RVVI->hasPolicyOperand())
      OS << "  Ops.push_back(ConstantInt::get(Ops.back()->getType(), "
            "TAIL_UNDISTURBED));\n";
    else if (RVVI->hasPassthruOperand()) {
      OS << "  Ops.push_back(llvm::UndefValue::get(ResultType));\n";
      OS << "  std::rotate(Ops.rbegin(), Ops.rbegin() + 1,  Ops.rend());\n";
    }
  }

  OS << "  IntrinsicTypes = {";
  ListSeparator LS;
  for (const auto &Idx : RVVI->getIntrinsicTypes()) {
    if (Idx == -1)
      OS << LS << "ResultType";
    else
      OS << LS << "Ops[" << Idx << "]->getType()";
  }

  // VL could be i64 or i32, need to encode it in IntrinsicTypes. VL is
  // always last operand.
  if (RVVI->hasVL())
    OS << ", Ops.back()->getType()";
  OS << "};\n";
  OS << "  break;\n";
}

//===----------------------------------------------------------------------===//
// RVVEmitter implementation
//===----------------------------------------------------------------------===//
void RVVEmitter::createHeader(raw_ostream &OS) {

  OS << "/*===---- riscv_vector.h - RISC-V V-extension RVVIntrinsics "
        "-------------------===\n"
        " *\n"
        " *\n"
        " * Part of the LLVM Project, under the Apache License v2.0 with LLVM "
        "Exceptions.\n"
        " * See https://llvm.org/LICENSE.txt for license information.\n"
        " * SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception\n"
        " *\n"
        " *===-----------------------------------------------------------------"
        "------===\n"
        " */\n\n";

  OS << "#ifndef __RISCV_VECTOR_H\n";
  OS << "#define __RISCV_VECTOR_H\n\n";

  OS << "#include <stdint.h>\n";
  OS << "#include <stddef.h>\n\n";

  OS << "#ifndef __riscv_vector\n";
  OS << "#error \"Vector intrinsics require the vector extension.\"\n";
  OS << "#endif\n\n";

  OS << "#ifdef __cplusplus\n";
  OS << "extern \"C\" {\n";
  OS << "#endif\n\n";

  OS << "#pragma clang riscv intrinsic vector\n\n";

  printHeaderCode(OS);

  auto printType = [&](auto T) {
    OS << "typedef " << T->getClangBuiltinStr() << " " << T->getTypeStr()
       << ";\n";
  };

  constexpr int Log2LMULs[] = {-3, -2, -1, 0, 1, 2, 3};
  // Print RVV boolean types.
  for (int Log2LMUL : Log2LMULs) {
    auto T = RVVType::computeType(BasicType::Int8, Log2LMUL, TypeProfile::Mask);
    if (T.hasValue())
      printType(T.getValue());
  }
  // Print RVV int/float types.
  for (char I : StringRef("csil")) {
    BasicType BT = ParseBasicType(I);
    for (int Log2LMUL : Log2LMULs) {
      auto T = RVVType::computeType(BT, Log2LMUL, TypeProfile::Vector);
      if (T.hasValue()) {
        printType(T.getValue());
        auto UT = RVVType::computeType(
            BT, Log2LMUL,
            TypeProfile(PrimitiveType::Vector, TypeModifier::UnsignedInteger));
        printType(UT.getValue());
      }
    }
  }
  OS << "#if defined(__riscv_zvfh)\n";
  for (int Log2LMUL : Log2LMULs) {
    auto T =
        RVVType::computeType(BasicType::Float16, Log2LMUL, TypeProfile::Vector);
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n";

  OS << "#if defined(__riscv_f)\n";
  for (int Log2LMUL : Log2LMULs) {
    auto T =
        RVVType::computeType(BasicType::Float32, Log2LMUL, TypeProfile::Vector);
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n";

  OS << "#if defined(__riscv_d)\n";
  for (int Log2LMUL : Log2LMULs) {
    auto T =
        RVVType::computeType(BasicType::Float64, Log2LMUL, TypeProfile::Vector);
    if (T.hasValue())
      printType(T.getValue());
  }
  OS << "#endif\n\n";

  OS << "#define __riscv_v_intrinsic_overloading 1\n";

  OS << "\n#ifdef __cplusplus\n";
  OS << "}\n";
  OS << "#endif // __cplusplus\n";
  OS << "#endif // __RISCV_VECTOR_H\n";
}

void RVVEmitter::createBuiltins(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);

  // Map to keep track of which builtin names have already been emitted.
  StringMap<RVVIntrinsic *> BuiltinMap;

  OS << "#if defined(TARGET_BUILTIN) && !defined(RISCVV_BUILTIN)\n";
  OS << "#define RISCVV_BUILTIN(ID, TYPE, ATTRS) TARGET_BUILTIN(ID, TYPE, "
        "ATTRS, \"zve32x\")\n";
  OS << "#endif\n";
  for (auto &Def : Defs) {
    auto P =
        BuiltinMap.insert(std::make_pair(Def->getBuiltinName(), Def.get()));
    if (!P.second) {
      // Verf that this would have produced the same builtin definition.
      if (P.first->second->hasBuiltinAlias() != Def->hasBuiltinAlias())
        PrintFatalError("Builtin with same name has different hasAutoDef");
      else if (!Def->hasBuiltinAlias() &&
               P.first->second->getBuiltinTypeStr() != Def->getBuiltinTypeStr())
        PrintFatalError("Builtin with same name has different type string");
      continue;
    }
    OS << "RISCVV_BUILTIN(__builtin_rvv_" << Def->getBuiltinName() << ",\"";
    if (!Def->hasBuiltinAlias())
      OS << Def->getBuiltinTypeStr();
    OS << "\", \"n\")\n";
  }
  OS << "#undef RISCVV_BUILTIN\n";
}

void RVVEmitter::createCodeGen(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);
  // IR name could be empty, use the stable sort preserves the relative order.
  llvm::stable_sort(Defs, [](const std::unique_ptr<RVVIntrinsic> &A,
                             const std::unique_ptr<RVVIntrinsic> &B) {
    return A->getIRName() < B->getIRName();
  });

  // Map to keep track of which builtin names have already been emitted.
  StringMap<RVVIntrinsic *> BuiltinMap;

  // Print switch body when the ir name or ManualCodegen changes from previous
  // iteration.
  RVVIntrinsic *PrevDef = Defs.begin()->get();
  for (auto &Def : Defs) {
    StringRef CurIRName = Def->getIRName();
    if (CurIRName != PrevDef->getIRName() ||
        (Def->getManualCodegen() != PrevDef->getManualCodegen())) {
      emitCodeGenSwitchBody(PrevDef, OS);
    }
    PrevDef = Def.get();

    auto P =
        BuiltinMap.insert(std::make_pair(Def->getBuiltinName(), Def.get()));
    if (P.second) {
      OS << "case RISCVVector::BI__builtin_rvv_" << Def->getBuiltinName()
         << ":\n";
      continue;
    }

    if (P.first->second->getIRName() != Def->getIRName())
      PrintFatalError("Builtin with same name has different IRName");
    else if (P.first->second->getManualCodegen() != Def->getManualCodegen())
      PrintFatalError("Builtin with same name has different ManualCodegen");
    else if (P.first->second->getNF() != Def->getNF())
      PrintFatalError("Builtin with same name has different NF");
    else if (P.first->second->isMasked() != Def->isMasked())
      PrintFatalError("Builtin with same name has different isMasked");
    else if (P.first->second->hasVL() != Def->hasVL())
      PrintFatalError("Builtin with same name has different hasVL");
    else if (P.first->second->getPolicyScheme() != Def->getPolicyScheme())
      PrintFatalError("Builtin with same name has different getPolicyScheme");
    else if (P.first->second->getIntrinsicTypes() != Def->getIntrinsicTypes())
      PrintFatalError("Builtin with same name has different IntrinsicTypes");
  }
  emitCodeGenSwitchBody(Defs.back().get(), OS);
  OS << "\n";
}

void RVVEmitter::createRVVIntrinsics(
    std::vector<std::unique_ptr<RVVIntrinsic>> &Out) {
  std::vector<Record *> RV = Records.getAllDerivedDefinitions("RVVBuiltin");
  for (auto *R : RV) {
    StringRef Name = R->getValueAsString("Name");
    StringRef Suffix = R->getValueAsString("Suffix");
    StringRef MangledName = R->getValueAsString("MangledName");
    StringRef OverloadedSuffix = R->getValueAsString("MangledSuffix");
    StringRef Prototypes = R->getValueAsString("Prototype");
    StringRef TypeRange = R->getValueAsString("TypeRange");
    bool HasMasked = R->getValueAsBit("HasMasked");
    bool HasMaskedOffOperand = R->getValueAsBit("HasMaskedOffOperand");
    bool HasVL = R->getValueAsBit("HasVL");
    Record *MaskedPolicyRecord = R->getValueAsDef("MaskedPolicy");
    PolicyScheme MaskedPolicy =
        static_cast<PolicyScheme>(MaskedPolicyRecord->getValueAsInt("Value"));
    Record *UnMaskedPolicyRecord = R->getValueAsDef("UnMaskedPolicy");
    PolicyScheme UnMaskedPolicy =
        static_cast<PolicyScheme>(UnMaskedPolicyRecord->getValueAsInt("Value"));
    bool HasUnMaskedOverloaded = R->getValueAsBit("HasUnMaskedOverloaded");
    std::vector<int64_t> Log2LMULList = R->getValueAsListOfInts("Log2LMUL");
    bool HasBuiltinAlias = R->getValueAsBit("HasBuiltinAlias");
    StringRef ManualCodegen = R->getValueAsString("ManualCodegen");
    StringRef MaskedManualCodegen = R->getValueAsString("MaskedManualCodegen");
    std::vector<int64_t> IntrinsicTypes =
        R->getValueAsListOfInts("IntrinsicTypes");
    std::vector<StringRef> RequiredFeatures =
        R->getValueAsListOfStrings("RequiredFeatures");
    StringRef IRName = R->getValueAsString("IRName");
    StringRef MaskedIRName = R->getValueAsString("MaskedIRName");
    unsigned NF = R->getValueAsInt("NF");

    // Parse prototype and create a list of primitive type with transformers
    // (operand) in Prototype. Prototype[0] is output operand.
    SmallVector<TypeProfile> Prototype = parsePrototypes(Prototypes);

    SmallVector<TypeProfile> Suffixtype = parsePrototypes(Suffix);
    SmallVector<TypeProfile> OverloadedSuffixtype =
        parsePrototypes(OverloadedSuffix);

    // Compute Builtin types
    SmallVector<TypeProfile> MaskedPrototype = Prototype;
    if (HasMasked) {
      // If HasMaskedOffOperand, insert result type as first input operand.
      if (HasMaskedOffOperand) {
        if (NF == 1) {
          MaskedPrototype.insert(MaskedPrototype.begin() + 1, Prototype[0]);
        } else {
          // Convert
          // (void, op0 address, op1 address, ...)
          // to
          // (void, op0 address, op1 address, ..., maskedoff0, maskedoff1, ...)
          TypeProfile MaskoffType = Prototype[1];
          MaskoffType.TM &= ~static_cast<uint8_t>(TypeModifier::Pointer);
          for (unsigned I = 0; I < NF; ++I)
            MaskedPrototype.insert(MaskedPrototype.begin() + NF + 1,
                                   MaskoffType);
        }
      }
      if (HasMaskedOffOperand && NF > 1) {
        // Convert
        // (void, op0 address, op1 address, ..., maskedoff0, maskedoff1, ...)
        // to
        // (void, op0 address, op1 address, ..., mask, maskedoff0, maskedoff1,
        // ...)
        MaskedPrototype.insert(MaskedPrototype.begin() + NF + 1,
                               TypeProfile::Mask);
      } else {
        // If HasMasked, insert TypeProfile:Mask as first input operand.
        MaskedPrototype.insert(MaskedPrototype.begin() + 1, TypeProfile::Mask);
      }
    }
    // If HasVL, append TypeProfile:VL to last operand
    if (HasVL) {
      Prototype.push_back(TypeProfile::VL);
      MaskedPrototype.push_back(TypeProfile::VL);
    }

    // Create Intrinsics for each type and LMUL.
    for (char I : TypeRange) {
      for (int Log2LMUL : Log2LMULList) {
        BasicType BT = ParseBasicType(I);
        Optional<RVVTypes> Types =
            RVVType::computeTypes(BT, Log2LMUL, NF, Prototype);
        // Ignored to create new intrinsic if there are any illegal types.
        if (!Types.hasValue())
          continue;

        auto SuffixStr = RVVIntrinsic::getSuffixStr(BT, Log2LMUL, Suffixtype);
        auto MangledSuffixStr =
            RVVIntrinsic::getSuffixStr(BT, Log2LMUL, OverloadedSuffixtype);
        // Create a unmasked intrinsic
        Out.push_back(std::make_unique<RVVIntrinsic>(
            Name, SuffixStr, MangledName, MangledSuffixStr, IRName,
            /*IsMasked=*/false, /*HasMaskedOffOperand=*/false, HasVL,
            UnMaskedPolicy, HasUnMaskedOverloaded, HasBuiltinAlias,
            ManualCodegen, Types.getValue(), IntrinsicTypes, RequiredFeatures,
            NF));
        if (HasMasked) {
          // Create a masked intrinsic
          Optional<RVVTypes> MaskTypes =
              RVVType::computeTypes(BT, Log2LMUL, NF, MaskedPrototype);
          Out.push_back(std::make_unique<RVVIntrinsic>(
              Name, SuffixStr, MangledName, MangledSuffixStr, MaskedIRName,
              /*IsMasked=*/true, HasMaskedOffOperand, HasVL, MaskedPolicy,
              HasUnMaskedOverloaded, HasBuiltinAlias, MaskedManualCodegen,
              MaskTypes.getValue(), IntrinsicTypes, RequiredFeatures, NF));
        }
      } // end for Log2LMULList
    }   // end for TypeRange

    // We don't emit vsetvli and vsetvlimax for SemaRecord.
    // They are handled by riscv_vector.h
    if (Name == "vsetvli" || Name == "vsetvlimax")
      continue;
    // Create SemaRecord
    SemaRecord SR;
    SR.Name = Name.str();
    SR.OverloadedName = MangledName.str();
    BasicType TypeRangeMask = BasicType::Unknown;
    for (char I : TypeRange) {
      TypeRangeMask |= ParseBasicType(I);
    }
    SR.TypeRangeMask = static_cast<unsigned>(TypeRangeMask);

    unsigned Log2LMULMask = 0;
    for (int Log2LMUL : Log2LMULList) {
      Log2LMULMask |= 1 << (Log2LMUL + 3);
    }

    SR.Log2LMULMask = Log2LMULMask;
    for (auto RequiredFeature : RequiredFeatures)
      SR.RequiredFeatures.push_back(RequiredFeature.str());

    SR.NF = NF;

    SR.Prototype = std::move(Prototype);

    if (HasMasked)
      SR.MaskedPrototype = std::move(MaskedPrototype);

    auto InitSuffixtype = [&](SmallVectorImpl<TypeProfile> &PS,
                              StringRef Prototypes) {
      PS = parsePrototypes(Prototypes);
    };

    InitSuffixtype(SR.Suffix, Suffix);
    InitSuffixtype(SR.OverloadedSuffix, OverloadedSuffix);

    SemaRecords.push_back(SR);
  }
}

void RVVEmitter::printHeaderCode(raw_ostream &OS) {
  std::vector<Record *> RVVHeaders =
      Records.getAllDerivedDefinitions("RVVHeader");
  for (auto *R : RVVHeaders) {
    StringRef HeaderCodeStr = R->getValueAsString("HeaderCode");
    OS << HeaderCodeStr.str();
  }
}

unsigned
RVVEmitter::GetSemaSignatureIndex(const SmallVector<TypeProfile> &Signature) {
  if (Signature.size() == 0)
    return 0;

  // Checking Signature already in table or not.
  if (Signature.size() < SemaSignatureTable.size()) {
    size_t Bound = SemaSignatureTable.size() - Signature.size() + 1;
    for (size_t Index = 0; Index < Bound; ++Index) {
      bool Match = true;
      for (size_t i = 0; i < Signature.size(); ++i) {
        if (Signature[i] != SemaSignatureTable[Index + i]) {
          Match = false;
          break;
        }
      }
      // Reuse if found in table.
      if (Match)
        return Index;
    }
  }

  // Insert Signature into SemaSignatureTable if not found in the table.
  size_t Index = SemaSignatureTable.size();
  for (const auto &Type : Signature) {
    SemaSignatureTable.push_back(Type);
  }
  return Index;
}

void RVVEmitter::ConstructSemaSignatureTable() {
  // Sort signature entries by length, let longer signature insert first, to
  // make it more possible to reuse table entries, that can reduce ~10% table
  // size.
  struct Compare {
    bool operator()(const SmallVector<TypeProfile> &A,
                    const SmallVector<TypeProfile> &B) {
      if (A.size() != B.size())
        return A.size() > B.size();

      size_t Len = A.size();
      for (size_t i = 0; i < Len; ++i) {
        if (A[i] != B[i])
          return A[i] > B[i];
      }

      return false;
    }
  };

  std::set<SmallVector<TypeProfile>, Compare> Signatures;
  auto InsertToSignatureSet = [&](const SmallVector<TypeProfile> &Signature) {
    if (Signature.empty())
      return;

    Signatures.insert(Signature);
  };

  for (const auto &SemaRecord : SemaRecords) {
    InsertToSignatureSet(SemaRecord.Prototype);
    InsertToSignatureSet(SemaRecord.MaskedPrototype);
    InsertToSignatureSet(SemaRecord.Suffix);
    InsertToSignatureSet(SemaRecord.OverloadedSuffix);
  }

  for (const auto &Sig : Signatures) {
    GetSemaSignatureIndex(Sig);
  }
}

void RVVEmitter::EmitSemaSignatureTable(raw_ostream &OS) {
  OS << "#ifdef DECL_SIGNATURE_TABLE\n";
  for (const auto &Sig : SemaSignatureTable)
    OS << "TypeProfile(" << static_cast<int>(Sig.PT) << ", "
       << static_cast<int>(Sig.VTM) << ", " << static_cast<int>(Sig.TM)
       << "),\n";
  OS << "#endif\n";
}

void RVVEmitter::EmitSemaRecords(raw_ostream &OS) {
  OS << "#ifdef DECL_INTRINSIC_RECORDS\n";
  for (const auto &SR : SemaRecords) {
    // Output *MUST* sync with RVVIntrinsicRecord in SemaRVVLookup.cpp.
    OS << "{"
       << "\"" << SR.Name << "\", ";

    if (SR.OverloadedName.empty())
      OS << "nullptr, ";
    else
      OS << "\"" << SR.OverloadedName << "\", ";

    OS << GetSemaSignatureIndex(SR.Prototype) << ", ";
    OS << GetSemaSignatureIndex(SR.MaskedPrototype) << ", ";
    OS << GetSemaSignatureIndex(SR.Suffix) << ", ";
    OS << GetSemaSignatureIndex(SR.OverloadedSuffix) << ", ";

    OS << SR.Prototype.size() << ", ";
    OS << SR.MaskedPrototype.size() << ", ";
    OS << SR.Suffix.size() << ", ";
    OS << SR.OverloadedSuffix.size() << ", ";

    if (SR.RequiredFeatures.empty())
      OS << "RVVRequire::None";
    else {
      ListSeparator LS(" | ");
      for (auto RequiredFeature : SR.RequiredFeatures)
        OS << LS << "RVVRequire::" << RequiredFeature;
    }
    OS << ",";

    OS << " /* Type Range Mask*/" << SR.TypeRangeMask << ", ";
    OS << " /* LMUL Mask = */ " << SR.Log2LMULMask << ", ";

    OS << SR.NF << "},\n";
  }
  OS << "#endif\n";
}

void RVVEmitter::createSema(raw_ostream &OS) {
  std::vector<std::unique_ptr<RVVIntrinsic>> Defs;
  createRVVIntrinsics(Defs);

  ConstructSemaSignatureTable();
  EmitSemaSignatureTable(OS);
  EmitSemaRecords(OS);
}

namespace clang {
void EmitRVVHeader(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createHeader(OS);
}

void EmitRVVBuiltins(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createBuiltins(OS);
}

void EmitRVVBuiltinCG(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createCodeGen(OS);
}

void EmitRVVBuiltinSema(RecordKeeper &Records, raw_ostream &OS) {
  RVVEmitter(Records).createSema(OS);
}

} // End namespace clang
