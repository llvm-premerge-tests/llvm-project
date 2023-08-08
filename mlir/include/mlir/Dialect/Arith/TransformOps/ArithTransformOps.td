//===- ArithTransformOps.td - Arith transformation ops ----*- tablegen -*--===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef ARITH_TRANSFORM_OPS
#define ARITH_TRANSFORM_OPS

include "mlir/Dialect/Transform/IR/TransformDialect.td"
include "mlir/Dialect/Transform/IR/TransformInterfaces.td"
include "mlir/Dialect/Transform/IR/TransformTypes.td"
include "mlir/IR/OpBase.td"

def ApplyArithToLLVMConversionPatternsOp : Op<Transform_Dialect,
    "apply_conversion_patterns.arith.arith_to_llvm",
    [DeclareOpInterfaceMethods<ConversionPatternDescriptorOpInterface,
                               ["verifyTypeConverter"]>]> {
  let description = [{
    Collects patterns that convert arith dialect ops to LLVM dialect ops. These
    patterns require an "LLVMTypeConverter".
  }];

  let assemblyFormat = "attr-dict";
}

#endif // ARITH_TRANSFORM_OPS
