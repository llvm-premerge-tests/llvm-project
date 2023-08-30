//===- LowerVectorShapeCast.cpp - Lower 'vector.shape_cast' operation -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements target-independent rewrites and utilities to lower the
// 'vector.shape_cast' operation.
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/Utils/IndexingUtils.h"
#include "mlir/Dialect/Utils/StructuredOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/Dialect/Vector/Transforms/LoweringPatterns.h"
#include "mlir/Dialect/Vector/Transforms/VectorRewritePatterns.h"
#include "mlir/Dialect/Vector/Utils/VectorUtils.h"
#include "mlir/IR/BuiltinAttributeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/ImplicitLocOpBuilder.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/VectorInterfaces.h"
#include "mlir/Support/LogicalResult.h"

#define DEBUG_TYPE "vector-shape-cast-lowering"

using namespace mlir;
using namespace mlir::vector;

namespace {
/// ShapeOp 2D -> 1D downcast serves the purpose of flattening 2-D to 1-D
/// vectors progressively on the way to target llvm.matrix intrinsics.
/// This iterates over the most major dimension of the 2-D vector and performs
/// rewrites into:
///   vector.extract from 2-D + vector.insert_strided_slice offset into 1-D
class ShapeCastOp2DDownCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();

    if (sourceVectorType.isScalable() || resultVectorType.isScalable())
      return failure();

    if (sourceVectorType.getRank() != 2 || resultVectorType.getRank() != 1)
      return failure();

    auto loc = op.getLoc();
    Value desc = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    unsigned mostMinorVectorSize = sourceVectorType.getShape()[1];
    for (int64_t i = 0, e = sourceVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::ExtractOp>(loc, op.getSource(), i);
      desc = rewriter.create<vector::InsertStridedSliceOp>(
          loc, vec, desc,
          /*offsets=*/i * mostMinorVectorSize, /*strides=*/1);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

/// ShapeOp 1D -> 2D upcast serves the purpose of unflattening 2-D from 1-D
/// vectors progressively.
/// This iterates over the most major dimension of the 2-D vector and performs
/// rewrites into:
///   vector.extract_strided_slice from 1-D + vector.insert into 2-D
/// Note that 1-D extract_strided_slice are lowered to efficient vector.shuffle.
class ShapeCastOp2DUpCastRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();

    if (sourceVectorType.isScalable() || resultVectorType.isScalable())
      return failure();

    if (sourceVectorType.getRank() != 1 || resultVectorType.getRank() != 2)
      return failure();

    auto loc = op.getLoc();
    Value desc = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    unsigned mostMinorVectorSize = resultVectorType.getShape()[1];
    for (int64_t i = 0, e = resultVectorType.getShape().front(); i != e; ++i) {
      Value vec = rewriter.create<vector::ExtractStridedSliceOp>(
          loc, op.getSource(), /*offsets=*/i * mostMinorVectorSize,
          /*sizes=*/mostMinorVectorSize,
          /*strides=*/1);
      desc = rewriter.create<vector::InsertOp>(loc, vec, desc, i);
    }
    rewriter.replaceOp(op, desc);
    return success();
  }
};

static void incIdx(llvm::MutableArrayRef<int64_t> idx, VectorType tp,
                   int dimIdx, int initialStep = 1) {
  int step = initialStep;
  for (int d = dimIdx; d >= 0; d--) {
    idx[d] += step;
    if (idx[d] >= tp.getDimSize(d)) {
      idx[d] = 0;
      step = 1;
    } else {
      break;
    }
  }
}

// We typically should not lower general shape cast operations into data
// movement instructions, since the assumption is that these casts are
// optimized away during progressive lowering. For completeness, however,
// we fall back to a reference implementation that moves all elements
// into the right place if we get here.
class ShapeCastOpRewritePattern : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {
    Location loc = op.getLoc();
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();

    if (sourceVectorType.isScalable() || resultVectorType.isScalable())
      return failure();

    // Special case 2D / 1D lowerings with better implementations.
    // TODO: make is ND / 1D to allow generic ND -> 1D -> MD.
    int64_t srcRank = sourceVectorType.getRank();
    int64_t resRank = resultVectorType.getRank();
    if ((srcRank == 2 && resRank == 1) || (srcRank == 1 && resRank == 2))
      return failure();

    // Generic ShapeCast lowering path goes all the way down to unrolled scalar
    // extract/insert chains.
    // TODO: consider evolving the semantics to only allow 1D source or dest and
    // drop this potentially very expensive lowering.
    // Compute number of elements involved in the reshape.
    int64_t numElts = 1;
    for (int64_t r = 0; r < srcRank; r++)
      numElts *= sourceVectorType.getDimSize(r);
    // Replace with data movement operations:
    //    x[0,0,0] = y[0,0]
    //    x[0,0,1] = y[0,1]
    //    x[0,1,0] = y[0,2]
    // etc., incrementing the two index vectors "row-major"
    // within the source and result shape.
    SmallVector<int64_t> srcIdx(srcRank);
    SmallVector<int64_t> resIdx(resRank);
    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));
    for (int64_t i = 0; i < numElts; i++) {
      if (i != 0) {
        incIdx(srcIdx, sourceVectorType, srcRank - 1);
        incIdx(resIdx, resultVectorType, resRank - 1);
      }

      Value extract;
      if (srcRank == 0) {
        // 0-D vector special case
        assert(srcIdx.empty() && "Unexpected indices for 0-D vector");
        extract = rewriter.create<vector::ExtractElementOp>(
            loc, op.getSourceVectorType().getElementType(), op.getSource());
      } else {
        extract =
            rewriter.create<vector::ExtractOp>(loc, op.getSource(), srcIdx);
      }

      if (resRank == 0) {
        // 0-D vector special case
        assert(resIdx.empty() && "Unexpected indices for 0-D vector");
        result = rewriter.create<vector::InsertElementOp>(loc, extract, result);
      } else {
        result =
            rewriter.create<vector::InsertOp>(loc, extract, result, resIdx);
      }
    }
    rewriter.replaceOp(op, result);
    return success();
  }
};

// A shape_cast lowering for scalable vectors with a single trailing scalable
// dimension. This is similar to the general shape_cast lowering but makes use
// of vector.scalable.insert and vector.scalable.extract to move elements a
// subvector at a time.
class ScalableShapeCastOpRewritePattern
    : public OpRewritePattern<vector::ShapeCastOp> {
public:
  using OpRewritePattern::OpRewritePattern;

  LogicalResult matchAndRewrite(vector::ShapeCastOp op,
                                PatternRewriter &rewriter) const override {

    Location loc = op.getLoc();
    auto sourceVectorType = op.getSourceVectorType();
    auto resultVectorType = op.getResultVectorType();
    auto srcRank = sourceVectorType.getRank();
    auto resRank = resultVectorType.getRank();

    // vector.scalable.insert/extract only accept 1D vectors, so only a trailing
    // scalable dim is supported.
    if (!isVectorTypeWithtrailingScalableDim(sourceVectorType) ||
        !isVectorTypeWithtrailingScalableDim(resultVectorType)) {
      return failure();
    }

    auto sourceMinScalableSize = sourceVectorType.getShape().back();
    auto resultMinScalableSize = resultVectorType.getShape().back();

    auto extractionSize =
        std::min(sourceMinScalableSize, resultMinScalableSize);

    int64_t minNumElts = 1;
    for (auto size : sourceVectorType.getShape())
      minNumElts *= size;

    SmallVector<int64_t> srcIdx(srcRank);
    SmallVector<int64_t> resIdx(resRank);

    auto extractionVectorType = VectorType::get(
        {extractionSize}, sourceVectorType.getElementType(), {true});

    Value result = rewriter.create<arith::ConstantOp>(
        loc, resultVectorType, rewriter.getZeroAttr(resultVectorType));

    Value currentResultScalableVector;
    for (int64_t i = 0; i < minNumElts; i += extractionSize) {

      Value extractedSubVector = op.getSource();
      if (srcRank != 1) {
        extractedSubVector = rewriter.create<vector::ExtractOp>(
            loc, op.getSource(), llvm::ArrayRef(srcIdx).drop_back());
      }
      if (extractionSize < sourceMinScalableSize) {
        extractedSubVector = rewriter.create<vector::ScalableExtractOp>(
            loc, extractionVectorType, extractedSubVector, srcIdx.back());
      }

      if (!currentResultScalableVector) {
        if (extractionSize == resultMinScalableSize) {
          currentResultScalableVector = extractedSubVector;
        } else if (resRank != 1) {
          currentResultScalableVector = rewriter.create<vector::ExtractOp>(
              loc, result, llvm::ArrayRef(resIdx).drop_back());
        } else {
          currentResultScalableVector = result;
        }
      }

      if (extractionSize < resultMinScalableSize) {
        currentResultScalableVector = rewriter.create<vector::ScalableInsertOp>(
            loc, extractedSubVector, currentResultScalableVector,
            resIdx.back());
      }

      if (resIdx.back() + extractionSize >= resultMinScalableSize) {
        // Will wrap to next scalable vector, insert complete 1D slice into
        // result.
        result = rewriter.create<vector::InsertOp>(
            loc, currentResultScalableVector, result,
            llvm::ArrayRef(resIdx).drop_back());
        currentResultScalableVector = {};
      }

      incIdx(srcIdx, sourceVectorType, srcRank - 1, extractionSize);
      incIdx(resIdx, resultVectorType, resRank - 1, extractionSize);
    }

    rewriter.replaceOp(op, result);
    return success();
  }

  static bool isVectorTypeWithtrailingScalableDim(VectorType type) {
    return type.getRank() >= 1 && type.getScalableDims().back() &&
           !llvm::is_contained(type.getScalableDims().drop_back(), true);
  }
};

} // namespace

void mlir::vector::populateVectorShapeCastLoweringPatterns(
    RewritePatternSet &patterns, PatternBenefit benefit) {
  patterns.add<ShapeCastOp2DDownCastRewritePattern,
               ShapeCastOp2DUpCastRewritePattern, ShapeCastOpRewritePattern,
               ScalableShapeCastOpRewritePattern>(patterns.getContext(),
                                                  benefit);
}
