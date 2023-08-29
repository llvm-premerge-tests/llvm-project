//===- ElementwiseToLinalg.cpp - conversion of elementwise to linalg ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Linalg/Passes.h"

#include "mlir/Dialect/Arith/Utils/Utils.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/Linalg/Utils/Utils.h"
#include "mlir/Transforms/DialectConversion.h"

#define DEBUG_TYPE "linalg-dma-address-space"

namespace mlir {
#define GEN_PASS_DEF_LINALGDMAADDRESSSPACE
#include "mlir/Dialect/Linalg/Passes.h.inc"
} // namespace mlir

using namespace mlir;
using namespace mlir::linalg;

namespace {
/// A permissive pass that tries to DMA linalg operations that accepts
/// MemRefs in one address (memory) space and to DMA it to another address
/// (memory) space. Currently only linalg generic operations are supported. This
/// is necessary due to HWs where operations are allowed to run (or can benefit
/// from) running the operations on faster address (memory) space.
class LinalgDMAAddressSpacePass
    : public impl::LinalgDMAAddressSpaceBase<LinalgDMAAddressSpacePass> {
public:
  LinalgDMAAddressSpacePass() = default;

  void runOnOperation() final;

private:
  // Tries to DMA the linalg generic operands to destination address (memory)
  // space.
  void tryDmaGenericOp(linalg::GenericOp genericOp);
  // Checks if an a given operand can be DMA to destination address (memory)
  // space.
  bool canDMAOperand(Value operand);
  // For inputs, allocates a memory before linalg generic operation and add
  // MemRef DMA Start and MemRef DMA Wait operations for the given operand. For
  // outputs, DMA Start and DMA Wait will be inserted after the linalg generic
  // operation.
  Value dmaOperand(linalg::GenericOp genericOp, Value operand, bool isInput);
};
} // end anonymous namespace

void LinalgDMAAddressSpacePass::runOnOperation() {
  func::FuncOp funcOp = getOperation();

  if (funcOp.isDeclaration()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Skipping declaration function " << funcOp.getName() << "\n");
    return;
  }

  if (srcAddrSpace == destAddrSpace) {
    funcOp.emitError("Source and destination address spaces must be different");
    return signalPassFailure();
  }

  funcOp->walk(
      [&](linalg::GenericOp genericOp) { tryDmaGenericOp(genericOp); });
}

void LinalgDMAAddressSpacePass::tryDmaGenericOp(linalg::GenericOp genericOp) {
  LLVM_DEBUG(llvm::dbgs() << "Converting operands address spaces of "
                          << genericOp << "\n");

  SmallVector<Value> newInputs;
  for (auto operand : genericOp.getInputs()) {
    Value newOperand;
    if (canDMAOperand(operand))
      newOperand = dmaOperand(genericOp, operand, true /*isInput*/);
    else
      newOperand = operand;

    newInputs.push_back(newOperand);
  }
  genericOp.getInputsMutable().assign(newInputs);

  SmallVector<Value> newOutputs;
  for (auto operand : genericOp.getOutputs()) {
    Value newOperand;
    if (canDMAOperand(operand))
      newOperand = dmaOperand(genericOp, operand, false /*isInput*/);
    else
      newOperand = operand;

    newOutputs.push_back(newOperand);
  }
  genericOp.getOutputsMutable().assign(newOutputs);
}

bool LinalgDMAAddressSpacePass::canDMAOperand(Value operand) {
  auto memRefType = dyn_cast<MemRefType>(operand.getType());
  if (!memRefType) {
    LLVM_DEBUG(llvm::dbgs() << "Only MemRef operands are supported for operand "
                            << operand << "\n");
    return false;
  }

  unsigned addrSpace = 0 /*default*/;
  Attribute spaceAttr = memRefType.getMemorySpace();
  if (spaceAttr) {
    addrSpace = spaceAttr.cast<mlir::IntegerAttr>().getInt();
  }

  if (addrSpace != srcAddrSpace) {
    LLVM_DEBUG(llvm::dbgs()
               << "Operand " << operand
               << " address space doesn't match source address space "
               << srcAddrSpace << "\n");
    return false;
  }
  if (!memRefType.hasStaticShape()) {
    LLVM_DEBUG(llvm::dbgs()
               << "Operand " << operand << " has dynamic dimensions\n");
    return false;
  }

  return true;
}

Value LinalgDMAAddressSpacePass::dmaOperand(linalg::GenericOp genericOp,
                                            Value operand, bool isInput) {
  OpBuilder builder(genericOp.getOperation());

  builder.setInsertionPoint(genericOp);
  auto loc = genericOp.getLoc();

  // Create a tag (single element 1-d memref) for the DMA.
  auto tagMemRefType = MemRefType::get({1}, builder.getIntegerType(32));
  auto tagBuffer = builder.create<memref::AllocOp>(loc, tagMemRefType);

  // Create a buffer in destination address space for the DMA.
  auto srcMemRefType = cast<MemRefType>(operand.getType());
  auto destMemRefType =
      MemRefType::get(srcMemRefType.getShape(), srcMemRefType.getElementType(),
                      AffineMap{}, builder.getI64IntegerAttr(destAddrSpace));
  auto destBufferAlloc = builder.create<memref::AllocOp>(loc, destMemRefType);

  auto dmaSource = isInput ? operand : destBufferAlloc.getResult();
  auto dmaDest = isInput ? destBufferAlloc.getResult() : operand;

  if (!isInput)
    // Four outputs only, the DMA should be after linalg generic
    builder.setInsertionPointAfter(genericOp);

  // DMA the whole source buffer.
  auto numElements = builder.create<arith::ConstantOp>(
      loc, builder.getIndexAttr((destMemRefType.getNumElements())));
  auto zero = builder.create<arith::ConstantOp>(loc, builder.getIndexAttr(0));
  auto rank = srcMemRefType.getRank();
  SmallVector<Value> sourceIndices(rank, zero);
  SmallVector<Value> destIndices(rank, zero);
  SmallVector<Value> tagIndices(1, zero);

  // Create async DMA and wait right before input / right after output.
  builder.create<memref::DmaStartOp>(loc, dmaSource, sourceIndices, dmaDest,
                                     destIndices, numElements, tagBuffer,
                                     tagIndices);
  builder.create<memref::DmaWaitOp>(loc, tagBuffer, tagIndices, numElements);

  return destBufferAlloc.getResult();
}

std::unique_ptr<Pass> mlir::createLinalgDMAAddressSpacePass() {
  return std::make_unique<LinalgDMAAddressSpacePass>();
}
