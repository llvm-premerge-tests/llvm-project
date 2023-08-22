//===- DynamicExtent.cpp - Dynamic extent related APIs ----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines APIs that track and query dynamic extent information.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"
#include "clang/AST/Expr.h"
#include "clang/Basic/LLVM.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/MemRegion.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/ProgramState.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SValBuilder.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SVals.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/SymbolManager.h"

REGISTER_MAP_WITH_PROGRAMSTATE(DynamicExtentMap, const clang::ento::MemRegion *,
                               clang::ento::DefinedOrUnknownSVal)

namespace clang {
namespace ento {

DefinedOrUnknownSVal getDynamicExtent(ProgramStateRef State,
                                      const MemRegion *MR, SValBuilder &SVB);

static bool isFlexibleArrayMember(const FieldDecl *FD) {
  const auto *RD = FD->getParent();
  if (!RD->hasFlexibleArrayMember())
    return false;

  const FieldDecl *LastFD = nullptr;
  for (const FieldDecl *It : RD->fields())
    LastFD = It;

  return FD == LastFD;
}

static std::optional<DefinedOrUnknownSVal>
getFlexibleArrayExtent(ProgramStateRef State, const MemRegion *MR,
                       SValBuilder &SVB) {
  const auto *FieldMR = dyn_cast<FieldRegion>(MR);
  if (nullptr == FieldMR)
    return std::nullopt;

  const FieldDecl *FD = FieldMR->getDecl();
  if (!isFlexibleArrayMember(FD))
    return std::nullopt;

  const RecordDecl *RD = FD->getParent();
  const MemRegion *BaseMR = FieldMR->getBaseRegion();
  auto BaseSize = getDynamicExtent(State, BaseMR, SVB);

  auto &C = SVB.getContext();
  uint64_t RecordSize = C.getTypeSizeInChars(C.getRecordType(RD)).getQuantity();
  SVal RecordSizeVal = SVB.makeIntVal(RecordSize, C.getSizeType());

  SVal BaseTooSmall =
      SVB.evalBinOp(State, BO_LT, BaseSize, RecordSizeVal, C.BoolTy);
  if (!BaseTooSmall.isUndef() &&
      State->assume(*BaseTooSmall.getAs<DefinedOrUnknownSVal>(), true))
    return std::nullopt;

  SVal FlexibleArrayExtent =
      SVB.evalBinOp(State, BO_Sub, BaseSize, RecordSizeVal, C.getSizeType());
  return FlexibleArrayExtent.getAs<DefinedOrUnknownSVal>();
}

DefinedOrUnknownSVal getDynamicExtent(ProgramStateRef State,
                                      const MemRegion *MR, SValBuilder &SVB) {
  MR = MR->StripCasts();

  if (const DefinedOrUnknownSVal *Size = State->get<DynamicExtentMap>(MR))
    return *Size;

  if (auto FlexibleArrayExtent = getFlexibleArrayExtent(State, MR, SVB))
    return *FlexibleArrayExtent;

  return MR->getMemRegionManager().getStaticSize(MR, SVB);
}

DefinedOrUnknownSVal getElementExtent(QualType Ty, SValBuilder &SVB) {
  return SVB.makeIntVal(SVB.getContext().getTypeSizeInChars(Ty).getQuantity(),
                        SVB.getArrayIndexType());
}

DefinedOrUnknownSVal getDynamicElementCount(ProgramStateRef State,
                                            const MemRegion *MR,
                                            SValBuilder &SVB,
                                            QualType ElementTy) {
  assert(MR != nullptr && "Not-null region expected");
  MR = MR->StripCasts();

  DefinedOrUnknownSVal Size = getDynamicExtent(State, MR, SVB);
  SVal ElementSize = getElementExtent(ElementTy, SVB);

  SVal ElementCount =
      SVB.evalBinOp(State, BO_Div, Size, ElementSize, SVB.getArrayIndexType());

  return ElementCount.castAs<DefinedOrUnknownSVal>();
}

SVal getDynamicExtentWithOffset(ProgramStateRef State, SVal BufV) {
  SValBuilder &SvalBuilder = State->getStateManager().getSValBuilder();
  const MemRegion *MRegion = BufV.getAsRegion();
  if (!MRegion)
    return UnknownVal();
  RegionOffset Offset = MRegion->getAsOffset();
  if (Offset.hasSymbolicOffset())
    return UnknownVal();
  const MemRegion *BaseRegion = MRegion->getBaseRegion();
  if (!BaseRegion)
    return UnknownVal();

  NonLoc OffsetInBytes = SvalBuilder.makeArrayIndex(
      Offset.getOffset() /
      MRegion->getMemRegionManager().getContext().getCharWidth());
  DefinedOrUnknownSVal ExtentInBytes =
      getDynamicExtent(State, BaseRegion, SvalBuilder);

  return SvalBuilder.evalBinOp(State, BinaryOperator::Opcode::BO_Sub,
                               ExtentInBytes, OffsetInBytes,
                               SvalBuilder.getArrayIndexType());
}

ProgramStateRef setDynamicExtent(ProgramStateRef State, const MemRegion *MR,
                                 DefinedOrUnknownSVal Size, SValBuilder &SVB) {
  MR = MR->StripCasts();

  if (Size.isUnknown())
    return State;

  return State->set<DynamicExtentMap>(MR->StripCasts(), Size);
}

} // namespace ento
} // namespace clang
