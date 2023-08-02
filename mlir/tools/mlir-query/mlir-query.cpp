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
#include "mlir/IR/Matchers.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Query/Matcher/Registry.h"
#include "mlir/Tools/mlir-query/MlirQueryMain.h"

using namespace mlir;

// This is needed because these matchers are defined as overloaded functions.
using HasOpAttrName = detail::AttrOpMatcher(StringRef);
using HasOpName = detail::NameOpMatcher(StringRef);
using IsConstantOp = detail::constant_op_matcher();

namespace test {
#ifdef MLIR_INCLUDE_TESTS
void registerTestDialect(DialectRegistry &);
#endif
} // namespace test

int main(int argc, char **argv) {

  DialectRegistry registry;
  registerAllDialects(registry);

  query::matcher::RegistryMaps registryData;

  // Matchers registered in alphabetical order for consistency:
  registryData.registerMatcher("hasOpAttrName",
                               static_cast<HasOpAttrName *>(m_Attr));
  registryData.registerMatcher("hasOpName", static_cast<HasOpName *>(m_Op));
  registryData.registerMatcher("isConstantOp",
                               static_cast<IsConstantOp *>(m_Constant));
  registryData.registerMatcher("isNegInfFloat", m_NegInfFloat);
  registryData.registerMatcher("isNegZeroFloat", m_NegZeroFloat);
  registryData.registerMatcher("isNonZero", m_NonZero);
  registryData.registerMatcher("isOne", m_One);
  registryData.registerMatcher("isOneFloat", m_OneFloat);
  registryData.registerMatcher("isPosInfFloat", m_PosInfFloat);
  registryData.registerMatcher("isPosZeroFloat", m_PosZeroFloat);
  registryData.registerMatcher("isZero", m_Zero);
  registryData.registerMatcher("isZeroFloat", m_AnyZeroFloat);

#ifdef MLIR_INCLUDE_TESTS
  test::registerTestDialect(registry);
#endif
  MLIRContext context(registry);

  return failed(mlirQueryMain(argc, argv, context, registryData));
}
