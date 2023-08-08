//===- VPlanPatternMatch.h - Match on the LLVM VPlan IR -------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file provides a simple and efficient mechanism for performing general
// tree-based pattern matches on the LLVM VPlan IR. It is heavily modelled on
// PatternMatch.h
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TRANSFORMS_VECTORIZE_VPLAN_PATTERN_MATH_H
#define LLVM_TRANSFORMS_VECTORIZE_VPLAN_PATTERN_MATH_H

#include "VPlan.h"
#include "llvm/IR/PatternMatch.h"

namespace llvm {
namespace PatternMatch {

inline bind_ty<VPValue> vp_Value(VPValue *&V) { return V; }

/// Match a specified Value*.
struct vp_specificval_ty {
  const VPValue *Val;

  vp_specificval_ty(const VPValue *V) : Val(V) {}

  template <typename ITy> bool match(ITy *V) { return V == Val; }
};

/// Match if we have a specific specified value.
inline vp_specificval_ty vp_Specific(const VPValue *V) { return V; };

struct vp_is_zero {
  template <typename ITy> bool match(ITy *V) {
    if (const Value *UV = V->getUnderlyingValue())
      return isa<Constant>(UV) && cast<Constant>(UV)->isNullValue();

    return false;
  }
};

/// Match any null constant or a vector with all elements equal to 0.
/// For vectors, this includes constants with undefined elements.
inline vp_is_zero vp_Zero() { return vp_is_zero(); }

template <typename LHS, unsigned Opcode> struct VPUnaryOp_match {
  LHS L;

  VPUnaryOp_match(const LHS &L) : L(L) {}

  template <typename T> bool match(T *V) {
    auto *I = dyn_cast<VPInstruction>(V);

    if (!I || I->getOpcode() != Opcode)
      return false;

    return L.match(I->getOperand(0));
  }
};

template <typename LHS>
inline VPUnaryOp_match<LHS, VPInstruction::Not> vp_Not(const LHS &L) {
  return VPUnaryOp_match<LHS, VPInstruction::Not>(L);
}

template <typename LHS, typename RHS, unsigned Opcode, bool Commutable = false>
struct VPLogicalOp_match {
  LHS L;
  RHS R;

  VPLogicalOp_match(const LHS &L, const RHS &R) : L(L), R(R) {}

  template <typename T> bool match(T *V) {
    auto *I = dyn_cast<VPInstruction>(V);

    if (!I)
      return false;

    if (cast<VPInstruction>(V)->getOpcode() != Instruction::Select)
      return false;

    auto *Cond = cast<VPInstruction>(V)->getOperand(0);
    auto *TVal = cast<VPInstruction>(V)->getOperand(1);
    auto *FVal = cast<VPInstruction>(V)->getOperand(2);

    // TODO: How best can we know the operation is on i1 types?

    if (Opcode == Instruction::And)
      if (vp_Zero().match(FVal))
        return (L.match(Cond) && R.match(TVal)) ||
               (Commutable && L.match(TVal) && R.match(Cond));

    return false;
  }
};

/// Matches L && R either in the form of L & R or L ? R : false.
/// Note that the latter form is poison-blocking.
template <typename LHS, typename RHS>
inline VPLogicalOp_match<LHS, RHS, Instruction::And>
vp_LogicalAnd(const LHS &L, const RHS &R) {
  return VPLogicalOp_match<LHS, RHS, Instruction::And>(L, R);
}

/// Matches L && R where L and R are arbitrary values.
inline auto vp_LogicalAnd() { return vp_LogicalAnd(m_Value(), m_Value()); }

/// Matches L && R with LHS and RHS in either order.
template <typename LHS, typename RHS>
inline VPLogicalOp_match<LHS, RHS, Instruction::And, true>
vp_c_LogicalAnd(const LHS &L, const RHS &R) {
  return VPLogicalOp_match<LHS, RHS, Instruction::And, true>(L, R);
}

} // namespace PatternMatch
} // namespace llvm

#endif // LLVM_TRANSFORMS_VECTORIZE_VPLAN_PATTERN_MATH_H
