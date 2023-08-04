//===- OptimizedBufferization.cpp - special cases for bufferization -------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
// In some special cases we can bufferize hlfir expressions in a more optimal
// way so as to avoid creating temporaries. This pass handles these. It should
// be run before the catch-all bufferization pass.
//
// This requires constant subexpression elimination to have already been run.
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Analysis/AliasAnalysis.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/HLFIRTools.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/HLFIR/HLFIRDialect.h"
#include "flang/Optimizer/HLFIR/HLFIROps.h"
#include "flang/Optimizer/HLFIR/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/Dominance.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "llvm/ADT/TypeSwitch.h"
#include <iterator>
#include <memory>
#include <optional>

namespace hlfir {
#define GEN_PASS_DEF_OPTIMIZEDBUFFERIZATION
#include "flang/Optimizer/HLFIR/Passes.h.inc"
} // namespace hlfir

#define DEBUG_TYPE "opt-bufferization"

namespace {

/// This transformation should match in place modification of arrays.
/// It should match code of the form
/// %array = some.operation // array has shape %shape
/// %expr = hlfir.elemental %shape : [...] {
/// bb0(%arg0: index)
///   %0 = hlfir.designate %array(%arg0)
///   [...] // no other reads or writes to %array
///   hlfir.yield_element %element
/// }
/// hlfir.assign %expr to %array
/// hlfir.destroy %expr
///
/// In this case, it is safe to turn the elemental into a do loop and modify
/// elements of %array in place without creating an extra temporary for the
/// elemental. Note that %array should dominate %expr, with the same value
/// used for the assignment. This indicates that CSE was able to prove there
/// were no modifications to the array between %array and the assignment
/// (including during the elemental). We must check that there are no reads
/// from the array at indexes which might conflict with the assignment. For
/// now we will keep that strict and say that all reads must be at the
/// elemental index (it is probably safe to read from higher indices if
/// lowering to an ordered loop).
class ElementalAssignBufferization
    : public mlir::OpRewritePattern<hlfir::ElementalOp> {
private:
  struct MatchInfo {
    mlir::Value array;
    hlfir::AssignOp assign;
    hlfir::DestroyOp destroy;
  };
  /// determines if the transformation can be applied to this elemental
  static std::optional<MatchInfo> findMatch(hlfir::ElementalOp elemental);

public:
  using mlir::OpRewritePattern<hlfir::ElementalOp>::OpRewritePattern;

  mlir::LogicalResult
  matchAndRewrite(hlfir::ElementalOp elemental,
                  mlir::PatternRewriter &rewriter) const override;
};

std::optional<ElementalAssignBufferization::MatchInfo>
ElementalAssignBufferization::findMatch(hlfir::ElementalOp elemental) {
  mlir::Operation::user_range users = elemental->getUsers();
  // the only uses of the elemental should be the assignment and the destroy
  if (std::distance(users.begin(), users.end()) != 2)
    return std::nullopt;

  MatchInfo match;
  for (mlir::Operation *user : users)
    mlir::TypeSwitch<mlir::Operation *, void>(user)
        .Case([&](hlfir::AssignOp op) { match.assign = op; })
        .Case([&](hlfir::DestroyOp op) { match.destroy = op; });

  if (!match.assign || !match.destroy)
    return std::nullopt;

  // the array is what the elemental is assigned into
  // TODO: this could be extended to also allow hlfir.expr by first bufferizing
  // the incoming expression
  match.array = match.assign.getLhs();
  mlir::Type arrayType = mlir::dyn_cast<fir::SequenceType>(
      fir::unwrapPassByRefType(match.array.getType()));
  if (!arrayType)
    return std::nullopt;

  // require that the array elements are trivial
  // TODO: this is just to make the pass easier to think about. Not an inherent
  // limitation
  mlir::Type eleTy = hlfir::getFortranElementType(arrayType);
  if (!fir::isa_trivial(eleTy)) {
    return std::nullopt;
  }

  // does the array dominate the elmemental?
  mlir::DominanceInfo domInfo;
  // the array value can have no defining op if it is a block argument
  if (match.array.getDefiningOp())
    if (!domInfo.properlyDominates(match.array.getDefiningOp(), elemental))
      return std::nullopt;

  // the array must have the same shape as the elemental. CSE should have
  // deduplicated the fir.shape operations where they are provably the same
  // so we just have to check for the same ssa value
  // TODO: add more ways of getting the shape of the array
  mlir::Value arrayShape;
  if (match.array.getDefiningOp())
    arrayShape =
        mlir::TypeSwitch<mlir::Operation *, mlir::Value>(
            match.array.getDefiningOp())
            .Case([](hlfir::DesignateOp designate) {
              return designate.getShape();
            })
            .Case([](hlfir::DeclareOp declare) { return declare.getShape(); })
            .Default([](mlir::Operation *) { return mlir::Value{}; });
  if (!arrayShape) {
    LLVM_DEBUG(llvm::dbgs() << "Can't get shape of " << match.array << " at "
                            << elemental->getLoc() << "\n");
    return std::nullopt;
  }
  if (arrayShape != elemental.getShape())
    return std::nullopt;

  // check that there are no reads from the array other than at the elemental
  // index
  for (mlir::Operation *user : match.array.getUsers()) {
    if (user == match.assign)
      continue;
    if (domInfo.properlyDominates(user, elemental))
      // this use comes before the elemental so we don't care
      continue;
    if (domInfo.properlyDominates(match.assign, user))
      // this use comes after the assign so we don't care
      continue;
    if (user->getParentOp() == elemental) {
      // use inside of the elemental. The only thing that is okay is a
      // hlfir.designate using the elemental index
      hlfir::DesignateOp designate = mlir::dyn_cast<hlfir::DesignateOp>(user);
      if (!designate) {
        LLVM_DEBUG(llvm::dbgs() << "Non-designate use of match.array: " << *user
                                << " for " << elemental.getLoc() << "\n");
        return std::nullopt;
      }
      auto indices = designate.getIndices();
      auto elementalIndices = elemental.getIndices();
      if (indices.size() != elementalIndices.size()) {
        LLVM_DEBUG(llvm::dbgs() << "Mismatched designate indices: " << designate
                                << " for " << elemental.getLoc() << "\n");
        return std::nullopt;
      }
      if (!std::equal(indices.begin(), indices.end(), elementalIndices.begin(),
                      elementalIndices.end())) {
        LLVM_DEBUG(llvm::dbgs() << "Mismatched designate indices: " << designate
                                << " for " << elemental.getLoc() << "\n");
        return std::nullopt;
      }
      continue;
    }
    return std::nullopt;
  }

  fir::AliasAnalysis aliasAnalysis;
  // check for other memory accesses inside the elemental body
  for (mlir::Operation &op : *elemental.getBody()) {
    std::optional<llvm::SmallVector<mlir::MemoryEffects::EffectInstance>>
        effects = mlir::getEffectsRecursively(&op);
    // if we can't get effects for that operation then assume the worst case
    if (!effects) {
      LLVM_DEBUG(llvm::dbgs() << "Unknown effects for " << op << " for "
                              << elemental.getLoc() << "\n");
      return std::nullopt;
    }

    for (const mlir::MemoryEffects::EffectInstance &effect : *effects) {
      bool error =
          mlir::TypeSwitch<mlir::MemoryEffects::Effect *, bool>(
              effect.getEffect())
              .Case<mlir::MemoryEffects::Allocate>([](auto _) { return false; })
              .Case<mlir::MemoryEffects::Free>([](auto _) { return false; })
              // are there any write effects which might alias with the array?
              .Case<mlir::MemoryEffects::Write>([&](auto write) -> bool {
                LLVM_DEBUG(llvm::dbgs() << "Found write in " << op << " for "
                                        << elemental.getLoc() << "\n");
                return !aliasAnalysis.alias(match.array, effect.getValue())
                            .isNo();
              })
              // don't allow aliased reads
              .Case<mlir::MemoryEffects::Read>([&](auto read) -> bool {
                mlir::Value val = effect.getValue();
                if (val == match.array)
                  // already checked all uses of array
                  return false;

                // allow:
                // %ref = hlfir.designate %array(%index)
                // %val = fir.load %ref
                if (auto designate = val.getDefiningOp<hlfir::DesignateOp>()) {
                  // already checked all uses of array
                  if (designate.getMemref() == match.array)
                    return false;

                  // if the designate is into an array that definately doesn't
                  // alias match.array, then this should be safe. Alias analysis
                  // doesn't currently follow fir.load or understand
                  // hlfir.designate so it can't do this itself
                  if (aliasAnalysis.alias(match.array, designate.getMemref())
                          .isNo())
                    return false;
                }

                LLVM_DEBUG(llvm::dbgs() << "Found read in " << op << " for "
                                        << elemental.getLoc() << "\n");
                return !aliasAnalysis.alias(match.array, val).isNo();
              })
              .Default([](auto _) {
                llvm_unreachable("Unknown memory effect");
                return true;
              });
      if (error)
        return std::nullopt;
    }
  }

  return match;
}

mlir::LogicalResult ElementalAssignBufferization::matchAndRewrite(
    hlfir::ElementalOp elemental, mlir::PatternRewriter &rewriter) const {
  std::optional<MatchInfo> match = findMatch(elemental);
  if (!match)
    return rewriter.notifyMatchFailure(
        elemental, "cannot prove safety of ElementalAssignBufferization");

  mlir::Location loc = elemental->getLoc();
  fir::FirOpBuilder builder(rewriter, elemental.getOperation());
  auto extents = hlfir::getIndexExtents(loc, builder, elemental.getShape());

  // Generate a loop nest looping around the hlfir.elemental shape and clone
  // hlfir.elemental region inside the inner loop
  hlfir::LoopNest loopNest =
      hlfir::genLoopNest(loc, builder, extents, !elemental.isOrdered());
  builder.setInsertionPointToStart(loopNest.innerLoop.getBody());
  auto yield = hlfir::inlineElementalOp(loc, builder, elemental,
                                        loopNest.oneBasedIndices);
  hlfir::Entity elementValue{yield.getElementValue()};
  rewriter.eraseOp(yield);

  // Assign the element value to the array element for this iteration.
  auto arrayElement = hlfir::getElementAt(
      loc, builder, hlfir::Entity{match->array}, loopNest.oneBasedIndices);
  builder.create<hlfir::AssignOp>(
      loc, elementValue, arrayElement, /*realloc=*/false,
      /*keep_lhs_length_if_realloc=*/false, /*temporary_lhs=*/true);

  rewriter.eraseOp(match->assign);
  rewriter.eraseOp(match->destroy);
  rewriter.eraseOp(elemental);
  return mlir::success();
}

class OptimizedBufferizationPass
    : public hlfir::impl::OptimizedBufferizationBase<
          OptimizedBufferizationPass> {
public:
  void runOnOperation() override {
    mlir::func::FuncOp func = getOperation();
    mlir::MLIRContext *context = &getContext();

    mlir::GreedyRewriteConfig config;
    // Prevent the pattern driver from merging blocks
    config.enableRegionSimplification = false;

    mlir::RewritePatternSet patterns(context);
    patterns.insert<ElementalAssignBufferization>(context);

    if (mlir::failed(mlir::applyPatternsAndFoldGreedily(
            func, std::move(patterns), config))) {
      mlir::emitError(func.getLoc(),
                      "failure in HLFIR optimized bufferization");
      signalPassFailure();
    }
  }
};
} // namespace

std::unique_ptr<mlir::Pass> hlfir::createOptimizedBufferizationPass() {
  return std::make_unique<OptimizedBufferizationPass>();
}
