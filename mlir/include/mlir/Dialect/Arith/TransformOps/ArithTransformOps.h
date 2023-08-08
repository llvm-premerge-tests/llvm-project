//===- ArithTransformOps.h - Arith transformation ops -----------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_ARITH_TRANSFORMOPS_ARITHTRANSFORMOPS_H
#define MLIR_DIALECT_ARITH_TRANSFORMOPS_ARITHTRANSFORMOPS_H

#include "mlir/Bytecode/BytecodeOpInterface.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"
#include "mlir/IR/OpImplementation.h"

#define GET_OP_CLASSES
#include "mlir/Dialect/Arith/TransformOps/ArithTransformOps.h.inc"

namespace mlir {
class DialectRegistry;

namespace arith {
void registerTransformDialectExtension(DialectRegistry &registry);
} // namespace arith
} // namespace mlir

#endif // MLIR_DIALECT_ARITH_TRANSFORMOPS_ARITHTRANSFORMOPS_H
