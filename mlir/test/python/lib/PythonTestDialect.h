//===- PythonTestDialect.h - PythonTest dialect definition ------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TEST_PYTHON_LIB_PYTHONTESTDIALECT_H
#define MLIR_TEST_PYTHON_LIB_PYTHONTESTDIALECT_H

#include "mlir/IR/Dialect.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/Interfaces/InferTypeOpInterface.h"
#include "llvm/ADT/StringExtras.h"

#include "PythonTestDialect.h.inc"

#define GET_OP_CLASSES
#include "PythonTestOps.h.inc"

#define GET_ATTRDEF_CLASSES
#include "PythonTestAttributes.h.inc"

#define GET_TYPEDEF_CLASSES
#include "PythonTestTypes.h.inc"

namespace python_test {
inline std::string stringifyTestBitEnumVerticalBar(llvm::APInt symbol) {
  auto val = static_cast<uint32_t>(symbol.getLimitedValue());
  assert(7u == (7u | val) && "invalid bits set in bit enum");
  ::llvm::SmallVector<::llvm::StringRef, 2> strs;

  if (1u == (1u & val))
    strs.push_back("user");

  if (2u == (2u & val))
    strs.push_back("group");

  if (4u == (4u & val))
    strs.push_back("other");
  return llvm::join(strs, " | ");
}
// ::mlir::ParseResult OpWithBareBitEnumVerticalBar::parse ...
// valueAttr =
// parser.getBuilder().getIntegerAttr(parser.getBuilder().getIntegerType(32),
// static_cast<uint32_t>(*attrOptional));

} // namespace python_test

#include "PythonTestEnums.h.inc"

#endif // MLIR_TEST_PYTHON_LIB_PYTHONTESTDIALECT_H
