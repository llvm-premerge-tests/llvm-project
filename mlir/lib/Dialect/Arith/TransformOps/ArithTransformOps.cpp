//===- ArithTransformOps.cpp - Implementation of Arith transform ops ------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/Arith/TransformOps/ArithTransformOps.h"

#include "mlir/Conversion/ArithToLLVM/ArithToLLVM.h"
#include "mlir/Conversion/LLVMCommon/TypeConverter.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/Dialect/Transform/IR/TransformOps.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Apply...ConversionPatternsOp
//===----------------------------------------------------------------------===//

void transform::ApplyArithToLLVMConversionPatternsOp::populatePatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns) {
  arith::populateArithToLLVMConversionPatterns(
      static_cast<LLVMTypeConverter &>(typeConverter), patterns);
}

LogicalResult
transform::ApplyArithToLLVMConversionPatternsOp::verifyTypeConverter(
    transform::TypeConverterBuilderOpInterface builder) {
  if (builder.getTypeConverterType() != "LLVMTypeConverter")
    return emitOpError("expected LLVMTypeConverter");
  return success();
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class ArithTransformDialectExtension
    : public transform::TransformDialectExtension<
          ArithTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareGeneratedDialect<LLVM::LLVMDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/Arith/TransformOps/ArithTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/Arith/TransformOps/ArithTransformOps.cpp.inc"

void mlir::arith::registerTransformDialectExtension(DialectRegistry &registry) {
  registry.addExtensions<ArithTransformDialectExtension>();
}
