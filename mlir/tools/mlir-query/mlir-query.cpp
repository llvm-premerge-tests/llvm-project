//===- mlir-query.cpp - MLIR Query Driver -------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a command line utility that queries a file from/to MLIR using one
// of the registered queries.
//
//===----------------------------------------------------------------------===//

#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Tools/mlir-query/MlirQueryMain.h"

using namespace mlir;

namespace test {
#ifdef MLIR_INCLUDE_TESTS
void registerTestDialect(DialectRegistry &);
#endif
} // namespace test

int main(int argc, char **argv) {

  DialectRegistry registry;
  registerAllDialects(registry);
#ifdef MLIR_INCLUDE_TESTS
  test::registerTestDialect(registry);
#endif
  MLIRContext context(registry);

  return failed(mlirQueryMain(argc, argv, context));
}
