//===- Transforms.h - Bufferization and related transforms ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TRANSFORMS_H
#define MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/Bufferization/Transforms/OneShotAnalysis.h"
#include "mlir/IR/Operation.h"

namespace mlir {
namespace bufferization {
class AnalysisState;
struct BufferizationStatistics;
class OneShotAnalysisState;
struct OneShotBufferizationOptions;

/// Try to eliminate tensor::EmptyOps inside `op`.
///
/// * `rewriteFunc` generates the replacement for the tensor::EmptyOp.
/// * Only tensor::EmptyOps that are anchored on a matching OpOperand as per
///   `anchorMatchFunc` are considered. "Anchored" means that there is a path
///   on the reverse SSA use-def chain, starting from the OpOperand and always
///   following the aliasing  OpOperand, that eventually ends at a single
///   tensor::EmptyOp.
LogicalResult eliminateEmptyTensors(RewriterBase &rewriter, Operation *op,
                                    OneShotAnalysisState &state);

/// Within the given operation, hoist buffers from loops where possible. See
/// "BufferLoopHoistingPass" for more information.
void hoistBuffersFromLoops(Operation *op);

/// Resolve RaW and other conflicts by inserting bufferization.alloc_tensor ops.
/// After applying this transform, the IR can be bufferized without inserting
/// additional buffer allocations.
LogicalResult insertTensorCopies(Operation *op,
                                 const OneShotBufferizationOptions &options,
                                 BufferizationStatistics *statistics = nullptr);

/// Resolve RaW and other conflicts by inserting bufferization.alloc_tensor ops.
/// After applying this transform, the IR can be bufferized without inserting
/// additional buffer allocations.
LogicalResult insertTensorCopies(Operation *op, const AnalysisState &state);

/// Populate patterns to lower tensor.empty ops to bufferization.alloc_tensor
/// ops.
void populateEmptyTensorToAllocTensorPattern(RewritePatternSet &patterns);

} // namespace bufferization
} // namespace mlir

#endif // MLIR_DIALECT_BUFFERIZATION_TRANSFORMS_TRANSFORMS_H
