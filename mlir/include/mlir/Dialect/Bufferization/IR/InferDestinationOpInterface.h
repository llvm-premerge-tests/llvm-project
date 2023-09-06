//===- InferDestinationOpInterface.h - Infer destination --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_BUFFERIZATION_IR_INFERDESTINATIONOPINTERFACE_H_
#define MLIR_DIALECT_BUFFERIZATION_IR_INFERDESTINATIONOPINTERFACE_H_

#include "mlir/IR/OpDefinition.h"

namespace mlir {
class OpBuilder;
} // namespace mlir

#include "mlir/Dialect/Bufferization/IR/InferDestinationOpInterface.h.inc"

#endif // MLIR_DIALECT_BUFFERIZATION_IR_INFERDESTINATIONOPINTERFACE_H_
