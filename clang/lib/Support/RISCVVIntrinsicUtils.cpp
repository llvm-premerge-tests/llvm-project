//===- RISCVVIntrinsicUtils.cpp - RISC-V Vector Intrinsic Utils -*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "clang/Support/RISCVVIntrinsicUtils.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/Optional.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/ADT/StringExtras.h"
#include "llvm/ADT/StringMap.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <numeric>

using namespace llvm;

namespace clang {
namespace RISCV {

const TypeProfile TypeProfile::Mask = TypeProfile(PrimitiveType::MaskVector);
const TypeProfile TypeProfile::VL = TypeProfile(PrimitiveType::SizeT);
const TypeProfile TypeProfile::Vector = TypeProfile(PrimitiveType::Vector);

// Concat BasicType, LMUL and Proto as key
static StringMap<RVVType> LegalTypes;
static StringSet<> IllegalTypes;

//===----------------------------------------------------------------------===//
// Type implementation
//===----------------------------------------------------------------------===//

LMULType::LMULType(int NewLog2LMUL) {
  // Check Log2LMUL is -3, -2, -1, 0, 1, 2, 3
  assert(NewLog2LMUL <= 3 && NewLog2LMUL >= -3 && "Bad LMUL number!");
  Log2LMUL = NewLog2LMUL;
}

std::string LMULType::str() const {
  if (Log2LMUL < 0)
    return "mf" + utostr(1ULL << (-Log2LMUL));
  return "m" + utostr(1ULL << Log2LMUL);
}

VScaleVal LMULType::getScale(unsigned ElementBitwidth) const {
  int Log2ScaleResult = 0;
  switch (ElementBitwidth) {
  default:
    break;
  case 8:
    Log2ScaleResult = Log2LMUL + 3;
    break;
  case 16:
    Log2ScaleResult = Log2LMUL + 2;
    break;
  case 32:
    Log2ScaleResult = Log2LMUL + 1;
    break;
  case 64:
    Log2ScaleResult = Log2LMUL;
    break;
  }
  // Illegal vscale result would be less than 1
  if (Log2ScaleResult < 0)
    return llvm::None;
  return 1 << Log2ScaleResult;
}

void LMULType::MulLog2LMUL(int log2LMUL) { Log2LMUL += log2LMUL; }

LMULType &LMULType::operator*=(uint32_t RHS) {
  assert(isPowerOf2_32(RHS));
  this->Log2LMUL = this->Log2LMUL + Log2_32(RHS);
  return *this;
}

RVVType::RVVType(BasicType BT, int Log2LMUL, const TypeProfile &prototype)
    : BT(BT), LMUL(LMULType(Log2LMUL)) {
  applyBasicType();
  applyModifier(prototype);
  Valid = verifyType();
  if (Valid) {
    initBuiltinStr();
    initTypeStr();
    if (isVector()) {
      initClangBuiltinStr();
    }
  }
}

// clang-format off
// boolean type are encoded the ratio of n (SEW/LMUL)
// SEW/LMUL | 1         | 2         | 4         | 8        | 16        | 32        | 64
// c type   | vbool64_t | vbool32_t | vbool16_t | vbool8_t | vbool4_t  | vbool2_t  | vbool1_t
// IR type  | nxv1i1    | nxv2i1    | nxv4i1    | nxv8i1   | nxv16i1   | nxv32i1   | nxv64i1

// type\lmul | 1/8    | 1/4      | 1/2     | 1       | 2        | 4        | 8
// --------  |------  | -------- | ------- | ------- | -------- | -------- | --------
// i64       | N/A    | N/A      | N/A     | nxv1i64 | nxv2i64  | nxv4i64  | nxv8i64
// i32       | N/A    | N/A      | nxv1i32 | nxv2i32 | nxv4i32  | nxv8i32  | nxv16i32
// i16       | N/A    | nxv1i16  | nxv2i16 | nxv4i16 | nxv8i16  | nxv16i16 | nxv32i16
// i8        | nxv1i8 | nxv2i8   | nxv4i8  | nxv8i8  | nxv16i8  | nxv32i8  | nxv64i8
// double    | N/A    | N/A      | N/A     | nxv1f64 | nxv2f64  | nxv4f64  | nxv8f64
// float     | N/A    | N/A      | nxv1f32 | nxv2f32 | nxv4f32  | nxv8f32  | nxv16f32
// half      | N/A    | nxv1f16  | nxv2f16 | nxv4f16 | nxv8f16  | nxv16f16 | nxv32f16
// clang-format on

bool RVVType::verifyType() const {
  if (ScalarType == Invalid)
    return false;
  if (isScalar())
    return true;
  if (!Scale.hasValue())
    return false;
  if (isFloat() && ElementBitwidth == 8)
    return false;
  unsigned V = Scale.getValue();
  switch (ElementBitwidth) {
  case 1:
  case 8:
    // Check Scale is 1,2,4,8,16,32,64
    return (V <= 64 && isPowerOf2_32(V));
  case 16:
    // Check Scale is 1,2,4,8,16,32
    return (V <= 32 && isPowerOf2_32(V));
  case 32:
    // Check Scale is 1,2,4,8,16
    return (V <= 16 && isPowerOf2_32(V));
  case 64:
    // Check Scale is 1,2,4,8
    return (V <= 8 && isPowerOf2_32(V));
  }
  return false;
}

void RVVType::initBuiltinStr() {
  assert(isValid() && "RVVType is invalid");
  switch (ScalarType) {
  case ScalarTypeKind::Void:
    BuiltinStr = "v";
    return;
  case ScalarTypeKind::Size_t:
    BuiltinStr = "z";
    if (IsImmediate)
      BuiltinStr = "I" + BuiltinStr;
    if (IsPointer)
      BuiltinStr += "*";
    return;
  case ScalarTypeKind::Ptrdiff_t:
    BuiltinStr = "Y";
    return;
  case ScalarTypeKind::UnsignedLong:
    BuiltinStr = "ULi";
    return;
  case ScalarTypeKind::SignedLong:
    BuiltinStr = "Li";
    return;
  case ScalarTypeKind::Boolean:
    assert(ElementBitwidth == 1);
    BuiltinStr += "b";
    break;
  case ScalarTypeKind::SignedInteger:
  case ScalarTypeKind::UnsignedInteger:
    switch (ElementBitwidth) {
    case 8:
      BuiltinStr += "c";
      break;
    case 16:
      BuiltinStr += "s";
      break;
    case 32:
      BuiltinStr += "i";
      break;
    case 64:
      BuiltinStr += "Wi";
      break;
    default:
      llvm_unreachable("Unhandled ElementBitwidth!");
    }
    if (isSignedInteger())
      BuiltinStr = "S" + BuiltinStr;
    else
      BuiltinStr = "U" + BuiltinStr;
    break;
  case ScalarTypeKind::Float:
    switch (ElementBitwidth) {
    case 16:
      BuiltinStr += "x";
      break;
    case 32:
      BuiltinStr += "f";
      break;
    case 64:
      BuiltinStr += "d";
      break;
    default:
      llvm_unreachable("Unhandled ElementBitwidth!");
    }
    break;
  default:
    llvm_unreachable("ScalarType is invalid!");
  }
  if (IsImmediate)
    BuiltinStr = "I" + BuiltinStr;
  if (isScalar()) {
    if (IsConstant)
      BuiltinStr += "C";
    if (IsPointer)
      BuiltinStr += "*";
    return;
  }
  BuiltinStr = "q" + utostr(Scale.getValue()) + BuiltinStr;
  // Pointer to vector types. Defined for segment load intrinsics.
  // segment load intrinsics have pointer type arguments to store the loaded
  // vector values.
  if (IsPointer)
    BuiltinStr += "*";
}

void RVVType::initClangBuiltinStr() {
  assert(isValid() && "RVVType is invalid");
  assert(isVector() && "Handle Vector type only");

  ClangBuiltinStr = "__rvv_";
  switch (ScalarType) {
  case ScalarTypeKind::Boolean:
    ClangBuiltinStr += "bool" + utostr(64 / Scale.getValue()) + "_t";
    return;
  case ScalarTypeKind::Float:
    ClangBuiltinStr += "float";
    break;
  case ScalarTypeKind::SignedInteger:
    ClangBuiltinStr += "int";
    break;
  case ScalarTypeKind::UnsignedInteger:
    ClangBuiltinStr += "uint";
    break;
  default:
    llvm_unreachable("ScalarTypeKind is invalid");
  }
  ClangBuiltinStr += utostr(ElementBitwidth) + LMUL.str() + "_t";
}

void RVVType::initTypeStr() {
  assert(isValid() && "RVVType is invalid");

  if (IsConstant)
    Str += "const ";

  auto getTypeString = [&](StringRef TypeStr) {
    if (isScalar())
      return Twine(TypeStr + Twine(ElementBitwidth) + "_t").str();
    return Twine("v" + TypeStr + Twine(ElementBitwidth) + LMUL.str() + "_t")
        .str();
  };

  switch (ScalarType) {
  case ScalarTypeKind::Void:
    Str = "void";
    return;
  case ScalarTypeKind::Size_t:
    Str = "size_t";
    if (IsPointer)
      Str += " *";
    return;
  case ScalarTypeKind::Ptrdiff_t:
    Str = "ptrdiff_t";
    return;
  case ScalarTypeKind::UnsignedLong:
    Str = "unsigned long";
    return;
  case ScalarTypeKind::SignedLong:
    Str = "long";
    return;
  case ScalarTypeKind::Boolean:
    if (isScalar())
      Str += "bool";
    else
      // Vector bool is special case, the formulate is
      // `vbool<N>_t = MVT::nxv<64/N>i1` ex. vbool16_t = MVT::4i1
      Str += "vbool" + utostr(64 / Scale.getValue()) + "_t";
    break;
  case ScalarTypeKind::Float:
    if (isScalar()) {
      if (ElementBitwidth == 64)
        Str += "double";
      else if (ElementBitwidth == 32)
        Str += "float";
      else if (ElementBitwidth == 16)
        Str += "_Float16";
      else
        llvm_unreachable("Unhandled floating type.");
    } else
      Str += getTypeString("float");
    break;
  case ScalarTypeKind::SignedInteger:
    Str += getTypeString("int");
    break;
  case ScalarTypeKind::UnsignedInteger:
    Str += getTypeString("uint");
    break;
  default:
    llvm_unreachable("ScalarType is invalid!");
  }
  if (IsPointer)
    Str += " *";
}

void RVVType::initShortStr() {
  switch (ScalarType) {
  case ScalarTypeKind::Boolean:
    assert(isVector());
    ShortStr = "b" + utostr(64 / Scale.getValue());
    return;
  case ScalarTypeKind::Float:
    ShortStr = "f" + utostr(ElementBitwidth);
    break;
  case ScalarTypeKind::SignedInteger:
    ShortStr = "i" + utostr(ElementBitwidth);
    break;
  case ScalarTypeKind::UnsignedInteger:
    ShortStr = "u" + utostr(ElementBitwidth);
    break;
  default:
    llvm_unreachable("Unhandled case!");
  }
  if (isVector())
    ShortStr += LMUL.str();
}

void RVVType::applyBasicType() {
  switch (BT) {
  case BasicType::Int8:
    ElementBitwidth = 8;
    ScalarType = ScalarTypeKind::SignedInteger;
    break;
  case BasicType::Int16:
    ElementBitwidth = 16;
    ScalarType = ScalarTypeKind::SignedInteger;
    break;
  case BasicType::Int32:
    ElementBitwidth = 32;
    ScalarType = ScalarTypeKind::SignedInteger;
    break;
  case BasicType::Int64:
    ElementBitwidth = 64;
    ScalarType = ScalarTypeKind::SignedInteger;
    break;
  case BasicType::Float16:
    ElementBitwidth = 16;
    ScalarType = ScalarTypeKind::Float;
    break;
  case BasicType::Float32:
    ElementBitwidth = 32;
    ScalarType = ScalarTypeKind::Float;
    break;
  case BasicType::Float64:
    ElementBitwidth = 64;
    ScalarType = ScalarTypeKind::Float;
    break;
  default:
    llvm_unreachable("Unhandled type code!");
  }
  assert(ElementBitwidth != 0 && "Bad element bitwidth!");
}

Optional<TypeProfile>
TypeProfile::parseTypeProfile(llvm::StringRef TypeProfileStr) {
  TypeProfile TP;
  PrimitiveType PT = PrimitiveType::Invalid;
  if (TypeProfileStr.empty())
    return TP;
  // Handle primitive type transformer
  auto PType = TypeProfileStr.back();
  switch (PType) {
  case 'e':
    PT = PrimitiveType::Scalar;
    break;
  case 'v':
    PT = PrimitiveType::Vector;
    break;
  case 'w':
    PT = PrimitiveType::Widening2XVector;
    break;
  case 'q':
    PT = PrimitiveType::Widening4XVector;
    break;
  case 'o':
    PT = PrimitiveType::Widening8XVector;
    break;
  case 'm':
    PT = PrimitiveType::MaskVector;
    break;
  case '0':
    PT = PrimitiveType::Void;
    break;
  case 'z':
    PT = PrimitiveType::SizeT;
    break;
  case 't':
    PT = PrimitiveType::Ptrdiff;
    break;
  case 'u':
    PT = PrimitiveType::UnsignedLong;
    break;
  case 'l':
    PT = PrimitiveType::SignedLong;
    break;
  default:
    llvm_unreachable("Illegal primitive type transformers!");
  }
  TP.PT = static_cast<uint8_t>(PT);
  TypeProfileStr = TypeProfileStr.drop_back();

  // Extract and compute complex type transformer. It can only appear one time.
  if (TypeProfileStr.startswith("(")) {
    size_t Idx = TypeProfileStr.find(')');
    assert(Idx != StringRef::npos);
    StringRef ComplexType = TypeProfileStr.slice(1, Idx);
    TypeProfileStr = TypeProfileStr.drop_front(Idx + 1);
    assert(!TypeProfileStr.contains('(') &&
           "Only allow one complex type transformer");

    auto ComplexTT = ComplexType.split(":");
    VectorTypeModifier VTM = VectorTypeModifier::NoModifier;
    if (ComplexTT.first == "Log2EEW") {
      uint32_t Log2EEW;
      if (ComplexTT.second.getAsInteger(10, Log2EEW)) {
        llvm_unreachable("Invalid Log2EEW value!");
        return None;
      }
      switch (Log2EEW) {
      case 3:
        VTM = VectorTypeModifier::Log2EEW3;
        break;
      case 4:
        VTM = VectorTypeModifier::Log2EEW4;
        break;
      case 5:
        VTM = VectorTypeModifier::Log2EEW5;
        break;
      case 6:
        VTM = VectorTypeModifier::Log2EEW6;
        break;
      default:
        llvm_unreachable("Invalid Log2EEW value, should be [3-6]");
        return None;
      }
    } else if (ComplexTT.first == "FixedSEW") {
      uint32_t NewSEW;
      if (ComplexTT.second.getAsInteger(10, NewSEW)) {
        llvm_unreachable("Invalid FixedSEW value!");
        return None;
      }
      switch (NewSEW) {
      case 8:
        VTM = VectorTypeModifier::FixedSEW8;
        break;
      case 16:
        VTM = VectorTypeModifier::FixedSEW16;
        break;
      case 32:
        VTM = VectorTypeModifier::FixedSEW32;
        break;
      case 64:
        VTM = VectorTypeModifier::FixedSEW64;
        break;
      default:
        llvm_unreachable("Invalid FixedSEW value, should be 8, 16, 32 or 64");
        return None;
      }
    } else if (ComplexTT.first == "LFixedLog2LMUL") {
      int32_t Log2LMUL;
      if (ComplexTT.second.getAsInteger(10, Log2LMUL)) {
        llvm_unreachable("Invalid LFixedLog2LMUL value!");
        return None;
      }
      switch (Log2LMUL) {
      case -3:
        VTM = VectorTypeModifier::LFixedLog2LMULN3;
        break;
      case -2:
        VTM = VectorTypeModifier::LFixedLog2LMULN2;
        break;
      case -1:
        VTM = VectorTypeModifier::LFixedLog2LMULN1;
        break;
      case 0:
        VTM = VectorTypeModifier::LFixedLog2LMUL0;
        break;
      case 1:
        VTM = VectorTypeModifier::LFixedLog2LMUL1;
        break;
      case 2:
        VTM = VectorTypeModifier::LFixedLog2LMUL2;
        break;
      case 3:
        VTM = VectorTypeModifier::LFixedLog2LMUL3;
        break;
      default:
        llvm_unreachable("Invalid LFixedLog2LMUL value, should be [-3, 3]");
        return None;
      }
    } else if (ComplexTT.first == "SFixedLog2LMUL") {
      int32_t Log2LMUL;
      if (ComplexTT.second.getAsInteger(10, Log2LMUL)) {
        llvm_unreachable("Invalid SFixedLog2LMUL value!");
        return None;
      }
      switch (Log2LMUL) {
      case -3:
        VTM = VectorTypeModifier::SFixedLog2LMULN3;
        break;
      case -2:
        VTM = VectorTypeModifier::SFixedLog2LMULN2;
        break;
      case -1:
        VTM = VectorTypeModifier::SFixedLog2LMULN1;
        break;
      case 0:
        VTM = VectorTypeModifier::SFixedLog2LMUL0;
        break;
      case 1:
        VTM = VectorTypeModifier::SFixedLog2LMUL1;
        break;
      case 2:
        VTM = VectorTypeModifier::SFixedLog2LMUL2;
        break;
      case 3:
        VTM = VectorTypeModifier::SFixedLog2LMUL3;
        break;
      default:
        llvm_unreachable("Invalid LFixedLog2LMUL value, should be [-3, 3]");
        return None;
      }

    } else {
      llvm_unreachable("Illegal complex type transformers!");
    }
    TP.VTM = static_cast<uint8_t>(VTM);
  }

  // Compute the remain type transformers
  TypeModifier TM = TypeModifier::NoModifier;
  for (char I : TypeProfileStr) {
    switch (I) {
    case 'P':
      if ((TM & TypeModifier::Const) == TypeModifier::Const)
        llvm_unreachable("'P' transformer cannot be used after 'C'");
      if ((TM & TypeModifier::Pointer) == TypeModifier::Pointer)
        llvm_unreachable("'P' transformer cannot be used twice");
      TM |= TypeModifier::Pointer;
      break;
    case 'C':
      TM |= TypeModifier::Const;
      break;
    case 'K':
      TM |= TypeModifier::Immediate;
      break;
    case 'U':
      TM |= TypeModifier::UnsignedInteger;
      break;
    case 'I':
      TM |= TypeModifier::SignedInteger;
      break;
    case 'F':
      TM |= TypeModifier::Float;
      break;
    case 'S':
      TM |= TypeModifier::LMUL1;
      break;
    default:
      llvm_unreachable("Illegal non-primitive type transformer!");
    }
  }
  TP.TM = static_cast<uint8_t>(TM);

  return TP;
}

void RVVType::applyModifier(const TypeProfile &Transformer) {
  // Handle primitive type transformer
  switch (static_cast<PrimitiveType>(Transformer.PT)) {
  case PrimitiveType::Scalar:
    Scale = 0;
    break;
  case PrimitiveType::Vector:
    Scale = LMUL.getScale(ElementBitwidth);
    break;
  case PrimitiveType::Widening2XVector:
    ElementBitwidth *= 2;
    LMUL *= 2;
    Scale = LMUL.getScale(ElementBitwidth);
    break;
  case PrimitiveType::Widening4XVector:
    ElementBitwidth *= 4;
    LMUL *= 4;
    Scale = LMUL.getScale(ElementBitwidth);
    break;
  case PrimitiveType::Widening8XVector:
    ElementBitwidth *= 8;
    LMUL *= 8;
    Scale = LMUL.getScale(ElementBitwidth);
    break;
  case PrimitiveType::MaskVector:
    ScalarType = ScalarTypeKind::Boolean;
    Scale = LMUL.getScale(ElementBitwidth);
    ElementBitwidth = 1;
    break;
  case PrimitiveType::Void:
    ScalarType = ScalarTypeKind::Void;
    break;
  case PrimitiveType::SizeT:
    ScalarType = ScalarTypeKind::Size_t;
    break;
  case PrimitiveType::Ptrdiff:
    ScalarType = ScalarTypeKind::Ptrdiff_t;
    break;
  case PrimitiveType::UnsignedLong:
    ScalarType = ScalarTypeKind::UnsignedLong;
    break;
  case PrimitiveType::SignedLong:
    ScalarType = ScalarTypeKind::SignedLong;
    break;
  case PrimitiveType::Invalid:
    ScalarType = ScalarTypeKind::Invalid;
    return;
  default:
    llvm_unreachable("Illegal primitive type transformers!");
  }

  switch (static_cast<VectorTypeModifier>(Transformer.VTM)) {
  case VectorTypeModifier::Log2EEW3:
    applyLog2EEW(3);
    break;
  case VectorTypeModifier::Log2EEW4:
    applyLog2EEW(4);
    break;
  case VectorTypeModifier::Log2EEW5:
    applyLog2EEW(5);
    break;
  case VectorTypeModifier::Log2EEW6:
    applyLog2EEW(6);
    break;
  case VectorTypeModifier::FixedSEW8:
    applyFixedSEW(8);
    break;
  case VectorTypeModifier::FixedSEW16:
    applyFixedSEW(16);
    break;
  case VectorTypeModifier::FixedSEW32:
    applyFixedSEW(32);
    break;
  case VectorTypeModifier::FixedSEW64:
    applyFixedSEW(64);
    break;
  case VectorTypeModifier::LFixedLog2LMULN3:
    applyFixedLog2LMUL(-3, /* LargerThan= */ true);
    break;
  case VectorTypeModifier::LFixedLog2LMULN2:
    applyFixedLog2LMUL(-2, /* LargerThan= */ true);
    break;
  case VectorTypeModifier::LFixedLog2LMULN1:
    applyFixedLog2LMUL(-1, /* LargerThan= */ true);
    break;
  case VectorTypeModifier::LFixedLog2LMUL0:
    applyFixedLog2LMUL(0, /* LargerThan= */ true);
    break;
  case VectorTypeModifier::LFixedLog2LMUL1:
    applyFixedLog2LMUL(1, /* LargerThan= */ true);
    break;
  case VectorTypeModifier::LFixedLog2LMUL2:
    applyFixedLog2LMUL(2, /* LargerThan= */ true);
    break;
  case VectorTypeModifier::LFixedLog2LMUL3:
    applyFixedLog2LMUL(3, /* LargerThan= */ true);
    break;
  case VectorTypeModifier::SFixedLog2LMULN3:
    applyFixedLog2LMUL(-3, /* LargerThan= */ false);
    break;
  case VectorTypeModifier::SFixedLog2LMULN2:
    applyFixedLog2LMUL(-2, /* LargerThan= */ false);
    break;
  case VectorTypeModifier::SFixedLog2LMULN1:
    applyFixedLog2LMUL(-1, /* LargerThan= */ false);
    break;
  case VectorTypeModifier::SFixedLog2LMUL0:
    applyFixedLog2LMUL(0, /* LargerThan= */ false);
    break;
  case VectorTypeModifier::SFixedLog2LMUL1:
    applyFixedLog2LMUL(1, /* LargerThan= */ false);
    break;
  case VectorTypeModifier::SFixedLog2LMUL2:
    applyFixedLog2LMUL(2, /* LargerThan= */ false);
    break;
  case VectorTypeModifier::SFixedLog2LMUL3:
    applyFixedLog2LMUL(3, /* LargerThan= */ false);
    break;
  case VectorTypeModifier::NoModifier:
    break;
  default:
    llvm_unreachable("Illegal vector type modifier!");
  }

  for (unsigned TypeModifierMaskShift = 0;
       TypeModifierMaskShift <= static_cast<unsigned>(TypeModifier::MaxOffset);
       ++TypeModifierMaskShift) {
    unsigned TypeModifierMask = 1 << TypeModifierMaskShift;
    if ((static_cast<unsigned>(Transformer.TM) & TypeModifierMask) !=
        TypeModifierMask)
      continue;
    switch (static_cast<TypeModifier>(TypeModifierMask)) {
    case TypeModifier::Pointer:
      IsPointer = true;
      break;
    case TypeModifier::Const:
      IsConstant = true;
      break;
    case TypeModifier::Immediate:
      IsImmediate = true;
      IsConstant = true;
      break;
    case TypeModifier::UnsignedInteger:
      ScalarType = ScalarTypeKind::UnsignedInteger;
      break;
    case TypeModifier::SignedInteger:
      ScalarType = ScalarTypeKind::SignedInteger;
      break;
    case TypeModifier::Float:
      ScalarType = ScalarTypeKind::Float;
      break;
    case TypeModifier::LMUL1:
      LMUL = LMULType(0);
      // Update ElementBitwidth need to update Scale too.
      Scale = LMUL.getScale(ElementBitwidth);
      break;
    default:
      llvm_unreachable("Unknown type modifier mask!");
    }
  }
}

void RVVType::applyLog2EEW(unsigned Log2EEW) {
  // update new elmul = (eew/sew) * lmul
  LMUL.MulLog2LMUL(Log2EEW - Log2_32(ElementBitwidth));
  // update new eew
  ElementBitwidth = 1 << Log2EEW;
  ScalarType = ScalarTypeKind::SignedInteger;
  Scale = LMUL.getScale(ElementBitwidth);
}

void RVVType::applyFixedSEW(unsigned NewSEW) {
  // Set invalid type if src and dst SEW are same.
  if (ElementBitwidth == NewSEW) {
    ScalarType = ScalarTypeKind::Invalid;
    return;
  }
  // Update new SEW
  ElementBitwidth = NewSEW;
  Scale = LMUL.getScale(ElementBitwidth);
}

void RVVType::applyFixedLog2LMUL(int Log2LMUL, bool LargerThan) {
  if (LargerThan) {
    if (Log2LMUL < LMUL.Log2LMUL) {
      ScalarType = ScalarTypeKind::Invalid;
      return;
    }
  } else {
    if (Log2LMUL > LMUL.Log2LMUL) {
      ScalarType = ScalarTypeKind::Invalid;
      return;
    }
  }
  // Update new LMUL
  LMUL = LMULType(Log2LMUL);
  Scale = LMUL.getScale(ElementBitwidth);
}

Optional<RVVTypes> RVVType::computeTypes(BasicType BT, int Log2LMUL,
                                         unsigned NF,
                                         ArrayRef<TypeProfile> PrototypeSeq) {
  // LMUL x NF must be less than or equal to 8.
  if ((Log2LMUL >= 1) && (1 << Log2LMUL) * NF > 8)
    return llvm::None;

  RVVTypes Types;
  for (const TypeProfile &Proto : PrototypeSeq) {
    auto T = computeType(BT, Log2LMUL, Proto);
    if (!T.hasValue())
      return llvm::None;
    // Record legal type index
    Types.push_back(T.getValue());
  }
  return Types;
}

Optional<RVVTypePtr> RVVType::computeType(BasicType BT, int Log2LMUL,
                                          TypeProfile Proto) {
  std::string Idx =
      Twine(Twine(static_cast<int>(BT)) + Twine(Log2LMUL) + Proto.IndexStr())
          .str();
  // Search first
  auto It = LegalTypes.find(Idx);
  if (It != LegalTypes.end())
    return &(It->second);
  if (IllegalTypes.count(Idx))
    return llvm::None;
  // Compute type and record the result.
  RVVType T(BT, Log2LMUL, Proto);
  if (T.isValid()) {
    // Record legal type index and value.
    LegalTypes.insert({Idx, T});
    return &(LegalTypes[Idx]);
  }
  // Record illegal type index.
  IllegalTypes.insert(Idx);
  return llvm::None;
}

//===----------------------------------------------------------------------===//
// RVVIntrinsic implementation
//===----------------------------------------------------------------------===//
RVVIntrinsic::RVVIntrinsic(
    StringRef NewName, StringRef Suffix, StringRef NewMangledName,
    StringRef MangledSuffix, StringRef IRName, bool IsMasked,
    bool HasMaskedOffOperand, bool HasVL, PolicyScheme Scheme,
    bool HasUnMaskedOverloaded, bool HasBuiltinAlias, StringRef ManualCodegen,
    const RVVTypes &OutInTypes, const std::vector<int64_t> &NewIntrinsicTypes,
    const std::vector<StringRef> &RequiredFeatures, unsigned NF)
    : IRName(IRName), IsMasked(IsMasked), HasVL(HasVL), Scheme(Scheme),
      HasUnMaskedOverloaded(HasUnMaskedOverloaded),
      HasBuiltinAlias(HasBuiltinAlias), ManualCodegen(ManualCodegen.str()),
      NF(NF) {

  // Init BuiltinName, Name and MangledName
  BuiltinName = NewName.str();
  Name = BuiltinName;
  if (NewMangledName.empty())
    MangledName = NewName.split("_").first.str();
  else
    MangledName = NewMangledName.str();
  if (!Suffix.empty())
    Name += "_" + Suffix.str();
  if (!MangledSuffix.empty())
    MangledName += "_" + MangledSuffix.str();
  if (IsMasked) {
    BuiltinName += "_m";
    Name += "_m";
  }

  // Init OutputType and InputTypes
  OutputType = OutInTypes[0];
  InputTypes.assign(OutInTypes.begin() + 1, OutInTypes.end());

  // IntrinsicTypes is unmasked TA version index. Need to update it
  // if there is merge operand (It is always in first operand).
  IntrinsicTypes = NewIntrinsicTypes;
  if ((IsMasked && HasMaskedOffOperand) ||
      (!IsMasked && hasPassthruOperand())) {
    for (auto &I : IntrinsicTypes) {
      if (I >= 0)
        I += NF;
    }
  }
}

std::string RVVIntrinsic::getBuiltinTypeStr() const {
  std::string S;
  S += OutputType->getBuiltinStr();
  for (const auto &T : InputTypes) {
    S += T->getBuiltinStr();
  }
  return S;
}

std::string
RVVIntrinsic::getSuffixStr(BasicType Type, int Log2LMUL,
                           const llvm::ArrayRef<TypeProfile> &TypeProfiles) {
  SmallVector<std::string> SuffixStrs;
  for (auto TP : TypeProfiles) {
    auto T = RVVType::computeType(Type, Log2LMUL, TP);
    SuffixStrs.push_back(T.getValue()->getShortStr());
  }
  return join(SuffixStrs, "_");
}

SmallVector<TypeProfile> parsePrototypes(StringRef Prototypes) {
  SmallVector<TypeProfile> TypeProfiles;
  const StringRef Primaries("evwqom0ztul");
  while (!Prototypes.empty()) {
    size_t Idx = 0;
    // Skip over complex prototype because it could contain primitive type
    // character.
    if (Prototypes[0] == '(')
      Idx = Prototypes.find_first_of(')');
    Idx = Prototypes.find_first_of(Primaries, Idx);
    assert(Idx != StringRef::npos);
    auto TP = TypeProfile::parseTypeProfile(Prototypes.slice(0, Idx + 1));
    if (!TP)
      llvm_unreachable("Error during parsing prototype.");
    TypeProfiles.push_back(*TP);
    Prototypes = Prototypes.drop_front(Idx + 1);
  }
  return std::move(TypeProfiles);
}

} // end namespace RISCV
} // end namespace clang
