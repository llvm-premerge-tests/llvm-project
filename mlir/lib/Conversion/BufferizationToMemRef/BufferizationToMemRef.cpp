//===- BufferizationToMemRef.cpp - Bufferization to MemRef conversion -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert Bufferization dialect to MemRef
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/BufferizationToMemRef/BufferizationToMemRef.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTBUFFERIZATIONTOMEMREF
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
/// The CloneOpConversion transforms all bufferization clone operations into
/// memref alloc and memref copy operations. In the dynamic-shape case, it also
/// emits additional dim and constant operations to determine the shape. This
/// conversion does not resolve memory leaks if it is used alone.
struct CloneOpConversion : public OpConversionPattern<bufferization::CloneOp> {
  using OpConversionPattern<bufferization::CloneOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(bufferization::CloneOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Check for unranked memref types which are currently not supported.
    Type type = op.getType();
    if (isa<UnrankedMemRefType>(type)) {
      return rewriter.notifyMatchFailure(
          op, "UnrankedMemRefType is not supported.");
    }
    MemRefType memrefType = cast<MemRefType>(type);
    MemRefLayoutAttrInterface layout;
    auto allocType =
        MemRefType::get(memrefType.getShape(), memrefType.getElementType(),
                        layout, memrefType.getMemorySpace());
    // Since this implementation always allocates, certain result types of the
    // clone op cannot be lowered.
    if (!memref::CastOp::areCastCompatible({allocType}, {memrefType}))
      return failure();

    // Transform a clone operation into alloc + copy operation and pay
    // attention to the shape dimensions.
    Location loc = op->getLoc();
    SmallVector<Value, 4> dynamicOperands;
    for (int i = 0; i < memrefType.getRank(); ++i) {
      if (!memrefType.isDynamicDim(i))
        continue;
      Value dim = rewriter.createOrFold<memref::DimOp>(loc, op.getInput(), i);
      dynamicOperands.push_back(dim);
    }

    // Allocate a memref with identity layout.
    Value alloc = rewriter.create<memref::AllocOp>(op->getLoc(), allocType,
                                                   dynamicOperands);
    // Cast the allocation to the specified type if needed.
    if (memrefType != allocType)
      alloc = rewriter.create<memref::CastOp>(op->getLoc(), memrefType, alloc);
    rewriter.replaceOp(op, alloc);
    rewriter.create<memref::CopyOp>(loc, op.getInput(), alloc);
    return success();
  }
};

/// The DeallocOpConversion transforms all bufferization dealloc operations into
/// memref dealloc operations potentially guarded by scf if operations.
/// Additionally, memref extract_aligned_pointer_as_index and arith operations
/// are inserted to compute the guard conditions. We distinguish multiple cases
/// to provide an overall more efficient lowering. In the general case, a helper
/// func is created to avoid quadratic code size explosion (relative to the
/// number of operands of the dealloc operation). For examples of each case,
/// refer to the documentation of the member functions of this class.
class DeallocOpConversion
    : public OpConversionPattern<bufferization::DeallocOp> {

  /// Lower a simple case avoiding the helper function. Ideally, static analysis
  /// can provide enough aliasing information to split the dealloc operations up
  /// into this simple case as much as possible before running this pass.
  ///
  /// Example:
  /// ```
  /// bufferization.dealloc (%arg0 : memref<2xf32>) if (%arg1)
  /// ```
  /// is lowered to
  /// ```
  /// scf.if %arg1 {
  ///   memref.dealloc %arg0 : memref<2xf32>
  /// }
  /// ```
  LogicalResult
  rewriteOneMemrefNoRetainCase(bufferization::DeallocOp op, OpAdaptor adaptor,
                               ConversionPatternRewriter &rewriter) const {
    rewriter.replaceOpWithNewOp<scf::IfOp>(
        op, adaptor.getConditions()[0], [&](OpBuilder &builder, Location loc) {
          builder.create<memref::DeallocOp>(loc, adaptor.getMemrefs()[0]);
          builder.create<scf::YieldOp>(loc);
        });
    return success();
  }

  /// Lowering that supports all features the dealloc operation has to offer. It
  /// computes the base pointer of each memref (as an index), stores them in a
  /// new memref and passes it to the helper function generated in
  /// 'buildDeallocationHelperFunction'. The results are stored in two memrefs
  /// of booleans passed as arguments. The first stores the condition under
  /// which the memref should be deallocated, the second one stores the
  /// ownership of the retained values which can be used to replace the result
  /// values of the `bufferization.dealloc` operation.
  ///
  /// Example:
  /// ```
  /// %0:2 = bufferization.dealloc (%arg0, %arg1 : memref<2xf32>, memref<5xf32>)
  ///                           if (%arg3, %arg4)
  ///                       retain (%arg2, %arg5 : memref<1xf32>, memref<2xf32>)
  /// ```
  /// lowers to (simplified):
  /// ```
  /// %c0 = arith.constant 0 : index
  /// %c1 = arith.constant 1 : index
  /// %alloc = memref.alloc() : memref<2xindex>
  /// %alloc_0 = memref.alloc() : memref<2xi1>
  /// %alloc_1 = memref.alloc() : memref<2xindex>
  /// %intptr = memref.extract_aligned_pointer_as_index %arg0
  /// memref.store %intptr, %alloc[%c0] : memref<2xindex>
  /// %intptr_2 = memref.extract_aligned_pointer_as_index %arg1
  /// memref.store %intptr_2, %alloc[%c1] : memref<2xindex>
  /// memref.store %arg3, %alloc_0[%c0] : memref<2xi1>
  /// memref.store %arg4, %alloc_0[%c1] : memref<2xi1>
  /// %intptr_5 = memref.extract_aligned_pointer_as_index %arg2
  /// memref.store %intptr_5, %alloc_1[%c0] : memref<2xindex>
  /// %intptr_7 = memref.extract_aligned_pointer_as_index %arg5
  /// memref.store %intptr_7, %alloc_1[%c1] : memref<2xindex>
  /// %cast = memref.cast %alloc : memref<2xindex> to memref<?xindex>
  /// %cast_9 = memref.cast %alloc_0 : memref<2xi1> to memref<?xi1>
  /// %cast_10 = memref.cast %alloc_1 : memref<2xindex> to memref<?xindex>
  /// %alloc_11 = memref.alloc() : memref<2xi1>
  /// %alloc_12 = memref.alloc() : memref<2xi1>
  /// %cast_13 = memref.cast %alloc_11 : memref<2xi1> to memref<?xi1>
  /// %cast_14 = memref.cast %alloc_12 : memref<2xi1> to memref<?xi1>
  /// call @dealloc_helper(%cast, %cast_10, %cast_9, %cast_13, %cast_14) : (...)
  /// %0 = memref.load %alloc_11[%c0] : memref<2xi1>
  /// %1 = memref.load %alloc_12[%c0] : memref<2xi1>
  /// scf.if %0 {
  ///   memref.dealloc %arg0 : memref<2xf32>
  /// }
  /// %2 = memref.load %alloc_11[%c1] : memref<2xi1>
  /// %3 = memref.load %alloc_12[%c1] : memref<2xi1>
  /// scf.if %2 {
  ///   memref.dealloc %arg1 : memref<5xf32>
  /// }
  /// memref.dealloc %alloc : memref<2xindex>
  /// memref.dealloc %alloc_1 : memref<2xindex>
  /// memref.dealloc %alloc_0 : memref<2xi1>
  /// memref.dealloc %alloc_11 : memref<2xi1>
  /// memref.dealloc %alloc_12 : memref<2xi1>
  /// // replace %0#0 with %1
  /// // replace %0#1 with %3
  /// ```
  LogicalResult rewriteGeneralCase(bufferization::DeallocOp op,
                                   OpAdaptor adaptor,
                                   ConversionPatternRewriter &rewriter) const {
    // Allocate two memrefs holding the base pointer indices of the list of
    // memrefs to be deallocated and the ones to be retained. These can then be
    // passed to the helper function and the for-loops can iterate over them.
    // Without storing them to memrefs, we could not use for-loops but only a
    // completely unrolled version of it, potentially leading to code-size
    // blow-up.
    Value toDeallocMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getMemrefs().size()},
                                     rewriter.getIndexType()));
    Value conditionMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getConditions().size()},
                                     rewriter.getI1Type()));
    Value toRetainMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getRetained().size()},
                                     rewriter.getIndexType()));

    auto getConstValue = [&](uint64_t value) -> Value {
      return rewriter.create<arith::ConstantOp>(op.getLoc(),
                                                rewriter.getIndexAttr(value));
    };

    // Extract the base pointers of the memrefs as indices to check for aliasing
    // at runtime.
    for (auto [i, toDealloc] : llvm::enumerate(adaptor.getMemrefs())) {
      Value memrefAsIdx =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(op.getLoc(),
                                                                  toDealloc);
      rewriter.create<memref::StoreOp>(op.getLoc(), memrefAsIdx,
                                       toDeallocMemref, getConstValue(i));
    }

    for (auto [i, cond] : llvm::enumerate(adaptor.getConditions()))
      rewriter.create<memref::StoreOp>(op.getLoc(), cond, conditionMemref,
                                       getConstValue(i));

    for (auto [i, toRetain] : llvm::enumerate(adaptor.getRetained())) {
      Value memrefAsIdx =
          rewriter.create<memref::ExtractAlignedPointerAsIndexOp>(op.getLoc(),
                                                                  toRetain);
      rewriter.create<memref::StoreOp>(op.getLoc(), memrefAsIdx, toRetainMemref,
                                       getConstValue(i));
    }

    // Cast the allocated memrefs to dynamic shape because we want only one
    // helper function no matter how many operands the bufferization.dealloc
    // has.
    Value castedDeallocMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType()),
        toDeallocMemref);
    Value castedCondsMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI1Type()),
        conditionMemref);
    Value castedRetainMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getIndexType()),
        toRetainMemref);

    Value deallocCondsMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getMemrefs().size()},
                                     rewriter.getI1Type()));
    Value retainCondsMemref = rewriter.create<memref::AllocOp>(
        op.getLoc(), MemRefType::get({(int64_t)adaptor.getRetained().size()},
                                     rewriter.getI1Type()));

    Value castedDeallocCondsMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI1Type()),
        deallocCondsMemref);
    Value castedRetainCondsMemref = rewriter.create<memref::CastOp>(
        op->getLoc(),
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI1Type()),
        retainCondsMemref);

    rewriter.create<func::CallOp>(
        op.getLoc(), deallocHelperFunc,
        SmallVector<Value>{castedDeallocMemref, castedRetainMemref,
                           castedCondsMemref, castedDeallocCondsMemref,
                           castedRetainCondsMemref});

    SmallVector<Value> replacements;
    for (unsigned i = 0, e = adaptor.getRetained().size(); i < e; ++i) {
      Value idxValue = getConstValue(i);
      Value shouldDealloc = rewriter.create<memref::LoadOp>(
          op.getLoc(), deallocCondsMemref, idxValue);
      Value ownership = rewriter.create<memref::LoadOp>(
          op.getLoc(), retainCondsMemref, idxValue);
      replacements.push_back(ownership);
      rewriter.create<scf::IfOp>(
          op.getLoc(), shouldDealloc, [&](OpBuilder &builder, Location loc) {
            builder.create<memref::DeallocOp>(loc, adaptor.getMemrefs()[i]);
            builder.create<scf::YieldOp>(loc);
          });
    }

    // Deallocate above allocated memrefs again to avoid memory leaks.
    // Deallocation will not be run on code after this stage.
    rewriter.create<memref::DeallocOp>(op.getLoc(), toDeallocMemref);
    rewriter.create<memref::DeallocOp>(op.getLoc(), toRetainMemref);
    rewriter.create<memref::DeallocOp>(op.getLoc(), conditionMemref);
    rewriter.create<memref::DeallocOp>(op.getLoc(), deallocCondsMemref);
    rewriter.create<memref::DeallocOp>(op.getLoc(), retainCondsMemref);

    rewriter.replaceOp(op, replacements);
    return success();
  }

public:
  DeallocOpConversion(MLIRContext *context, func::FuncOp deallocHelperFunc)
      : OpConversionPattern<bufferization::DeallocOp>(context),
        deallocHelperFunc(deallocHelperFunc) {}

  LogicalResult
  matchAndRewrite(bufferization::DeallocOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    // Lower the trivial case.
    if (adaptor.getMemrefs().empty())
      return rewriter.eraseOp(op), success();

    if (adaptor.getMemrefs().size() == 1 && adaptor.getRetained().empty())
      return rewriteOneMemrefNoRetainCase(op, adaptor, rewriter);

    return rewriteGeneralCase(op, adaptor, rewriter);
  }

  /// Build a helper function per compilation unit that can be called at
  /// bufferization dealloc sites to determine aliasing and ownership.
  ///
  /// The generated function takes two memrefs of indices and three memrefs of
  /// booleans as arguments:
  ///   * The first argument A should contain the result of the
  ///   extract_aligned_pointer_as_index operation applied to the memrefs to be
  ///   deallocated
  ///   * The second argument B should contain the result of the
  ///   extract_aligned_pointer_as_index operation applied to the memrefs to be
  ///   retained
  ///   * The third argument C should contain the conditions as passed directly
  ///   to the deallocation operation.
  ///   * The fourth argument D is used to pass results to the caller. Those
  ///   represent the condition under which the memref at the corresponding
  ///   position in A should be deallocated.
  ///   * The fifth argument E is used to pass results to the caller. It
  ///   provides the ownership value corresponding the the memref at the same
  ///   position in B
  ///
  /// This helper function is supposed to be called once for each
  /// `bufferization.dealloc` operation to determine the deallocation need and
  /// new ownership indicator for the retained values, but does not perform the
  /// deallocation itself.
  ///
  /// The first scf for loop zero-initializes the output memref for aggregation.
  /// The second scf for loop contains two more loops, the first of which
  /// computes whether the memref at the index given by the outer loop aliases
  /// with any memref in the list of retained memrefs.  The second nested loop
  /// additionally checks whether one of the previously deallocated memrefs
  /// aliases with the currently processed one.
  ///
  /// Generated code:
  /// ```
  /// func.func @dealloc_helper(
  ///     %arg0: memref<?xindex>,
  ///     %arg1: memref<?xindex>,
  ///     %arg2: memref<?xi1>,
  ///     %arg3: memref<?xi1>,
  ///     %arg4: memref<?xi1>) {
  ///   %c0 = arith.constant 0 : index
  ///   %c1 = arith.constant 1 : index
  ///   %true = arith.constant true
  ///   %false = arith.constant false
  ///   %dim = memref.dim %arg0, %c0 : memref<?xindex>
  ///   %dim_0 = memref.dim %arg1, %c0 : memref<?xindex>
  ///   scf.for %arg5 = %c0 to %dim_0 step %c1 {
  ///     memref.store %false, %arg4[%arg5] : memref<?xi1>
  ///   }
  ///   scf.for %arg5 = %c0 to %dim step %c1 {
  ///     %0 = memref.load %arg0[%arg5] : memref<?xindex>
  ///     %1 = memref.load %arg2[%arg5] : memref<?xi1>
  ///     %2 = scf.for %arg6 = %c0 to %dim_0 step %c1
  ///                  iter_args(%arg7 = %true) -> (i1) {
  ///       %5 = memref.load %arg1[%arg6] : memref<?xindex>
  ///       %6 = arith.cmpi eq, %5, %0 : index
  ///       scf.if %6 {
  ///         %9 = memref.load %arg4[%arg6] : memref<?xi1>
  ///         %10 = arith.ori %9, %1 : i1
  ///         memref.store %10, %arg4[%arg6] : memref<?xi1>
  ///       }
  ///       %7 = arith.cmpi ne, %5, %0 : index
  ///       %8 = arith.andi %arg7, %7 : i1
  ///       scf.yield %8 : i1
  ///     }
  ///     %3 = scf.for %arg6 = %c0 to %arg5 step %c1
  ///                  iter_args(%arg7 = %2) -> (i1) {
  ///       %5 = memref.load %arg0[%arg6] : memref<?xindex>
  ///       %6 = arith.cmpi ne, %5, %0 : index
  ///       %7 = arith.andi %arg7, %6 : i1
  ///       scf.yield %7 : i1
  ///     }
  ///     %4 = arith.andi %3, %1 : i1
  ///     memref.store %4, %arg3[%arg5] : memref<?xi1>
  ///   }
  ///   return
  /// }
  /// ```
  static func::FuncOp
  buildDeallocationHelperFunction(OpBuilder &builder, Location loc,
                                  SymbolTable &symbolTable) {
    Type indexMemrefType =
        MemRefType::get({ShapedType::kDynamic}, builder.getIndexType());
    Type boolMemrefType =
        MemRefType::get({ShapedType::kDynamic}, builder.getI1Type());
    SmallVector<Type> argTypes{indexMemrefType, indexMemrefType, boolMemrefType,
                               boolMemrefType, boolMemrefType};
    builder.clearInsertionPoint();

    // Generate the func operation itself.
    auto helperFuncOp = func::FuncOp::create(
        loc, "dealloc_helper", builder.getFunctionType(argTypes, {}));
    symbolTable.insert(helperFuncOp);
    auto &block = helperFuncOp.getFunctionBody().emplaceBlock();
    block.addArguments(argTypes, SmallVector<Location>(argTypes.size(), loc));

    builder.setInsertionPointToStart(&block);
    Value toDeallocMemref = helperFuncOp.getArguments()[0];
    Value toRetainMemref = helperFuncOp.getArguments()[1];
    Value conditionMemref = helperFuncOp.getArguments()[2];
    Value deallocCondsMemref = helperFuncOp.getArguments()[3];
    Value retainCondsMemref = helperFuncOp.getArguments()[4];

    // Insert some prerequisites.
    Value c0 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
    Value c1 = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(1));
    Value trueValue =
        builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(true));
    Value falseValue =
        builder.create<arith::ConstantOp>(loc, builder.getBoolAttr(false));
    Value toDeallocSize =
        builder.create<memref::DimOp>(loc, toDeallocMemref, c0);
    Value toRetainSize = builder.create<memref::DimOp>(loc, toRetainMemref, c0);

    builder.create<scf::ForOp>(
        loc, c0, toRetainSize, c1, std::nullopt,
        [&](OpBuilder &builder, Location loc, Value i, ValueRange iterArgs) {
          builder.create<memref::StoreOp>(loc, falseValue, retainCondsMemref,
                                          i);
          builder.create<scf::YieldOp>(loc);
        });

    builder.create<scf::ForOp>(
        loc, c0, toDeallocSize, c1, std::nullopt,
        [&](OpBuilder &builder, Location loc, Value outerIter,
            ValueRange iterArgs) {
          Value toDealloc =
              builder.create<memref::LoadOp>(loc, toDeallocMemref, outerIter);
          Value cond =
              builder.create<memref::LoadOp>(loc, conditionMemref, outerIter);

          // Build the first for loop that computes aliasing with retained
          // memrefs.
          Value noRetainAlias =
              builder
                  .create<scf::ForOp>(
                      loc, c0, toRetainSize, c1, trueValue,
                      [&](OpBuilder &builder, Location loc, Value i,
                          ValueRange iterArgs) {
                        Value retainValue = builder.create<memref::LoadOp>(
                            loc, toRetainMemref, i);
                        Value doesAlias = builder.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::eq, retainValue,
                            toDealloc);
                        builder.create<scf::IfOp>(
                            loc, doesAlias,
                            [&](OpBuilder &builder, Location loc) {
                              Value retainCondValue =
                                  builder.create<memref::LoadOp>(
                                      loc, retainCondsMemref, i);
                              Value aggregatedRetainCond =
                                  builder.create<arith::OrIOp>(
                                      loc, retainCondValue, cond);
                              builder.create<memref::StoreOp>(
                                  loc, aggregatedRetainCond, retainCondsMemref,
                                  i);
                              builder.create<scf::YieldOp>(loc);
                            });
                        Value doesntAlias = builder.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::ne, retainValue,
                            toDealloc);
                        Value yieldValue = builder.create<arith::AndIOp>(
                            loc, iterArgs[0], doesntAlias);
                        builder.create<scf::YieldOp>(loc, yieldValue);
                      })
                  .getResult(0);

          // Build the second for loop that adds aliasing with previously
          // deallocated memrefs.
          Value noAlias =
              builder
                  .create<scf::ForOp>(
                      loc, c0, outerIter, c1, noRetainAlias,
                      [&](OpBuilder &builder, Location loc, Value i,
                          ValueRange iterArgs) {
                        Value prevDeallocValue = builder.create<memref::LoadOp>(
                            loc, toDeallocMemref, i);
                        Value doesntAlias = builder.create<arith::CmpIOp>(
                            loc, arith::CmpIPredicate::ne, prevDeallocValue,
                            toDealloc);
                        Value yieldValue = builder.create<arith::AndIOp>(
                            loc, iterArgs[0], doesntAlias);
                        builder.create<scf::YieldOp>(loc, yieldValue);
                      })
                  .getResult(0);

          Value shouldDealoc =
              builder.create<arith::AndIOp>(loc, noAlias, cond);
          builder.create<memref::StoreOp>(loc, shouldDealoc, deallocCondsMemref,
                                          outerIter);
          builder.create<scf::YieldOp>(loc);
        });

    builder.create<func::ReturnOp>(loc);
    return helperFuncOp;
  }

private:
  func::FuncOp deallocHelperFunc;
};
} // namespace

namespace {
struct BufferizationToMemRefPass
    : public impl::ConvertBufferizationToMemRefBase<BufferizationToMemRefPass> {
  BufferizationToMemRefPass() = default;

  void runOnOperation() override {
    ModuleOp module = cast<ModuleOp>(getOperation());
    OpBuilder builder =
        OpBuilder::atBlockBegin(&module.getBodyRegion().front());
    SymbolTable symbolTable(module);

    // Build dealloc helper function if there are deallocs.
    func::FuncOp helperFuncOp;
    getOperation()->walk([&](bufferization::DeallocOp deallocOp) {
      if (deallocOp.getMemrefs().size() > 1 ||
          !deallocOp.getRetained().empty()) {
        helperFuncOp = DeallocOpConversion::buildDeallocationHelperFunction(
            builder, getOperation()->getLoc(), symbolTable);
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });

    RewritePatternSet patterns(&getContext());
    patterns.add<CloneOpConversion>(patterns.getContext());
    patterns.add<DeallocOpConversion>(patterns.getContext(), helperFuncOp);

    ConversionTarget target(getContext());
    target.addLegalDialect<memref::MemRefDialect, arith::ArithDialect,
                           scf::SCFDialect, func::FuncDialect>();
    target.addIllegalDialect<bufferization::BufferizationDialect>();

    if (failed(applyPartialConversion(getOperation(), target,
                                      std::move(patterns))))
      signalPassFailure();
  }
};
} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::createBufferizationToMemRefPass() {
  return std::make_unique<BufferizationToMemRefPass>();
}
