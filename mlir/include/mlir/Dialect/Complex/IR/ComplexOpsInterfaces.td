//===-- ComplexOpsInterfaces.td - complex op interfaces ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is the interfaces definition file for complex dialect.
//
//===----------------------------------------------------------------------===//

#ifndef COMPLEX_OPS_INTERFACES
#define COMPLEX_OPS_INTERFACES

include "mlir/IR/OpBase.td"
include "mlir/Dialect/Complex/IR/ComplexBase.td"

def ComplexFastMathInterface : OpInterface<"ComplexFastMathInterface"> {
  let description = [{
    Access to operation fastmath flags.
  }];

  let cppNamespace = "::mlir::complex";

  let methods = [
    InterfaceMethod<
      /*desc=*/        "Returns a FastMathFlagsAttr attribute for the operation",
      /*returnType=*/  "::mlir::arith::FastMathFlagsAttr",
      /*methodName=*/  "getFastMathFlagsAttr",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        ConcreteOp op = cast<ConcreteOp>(this->getOperation());
        return op.getFastmathAttr();
      }]
      >,
    StaticInterfaceMethod<
      /*desc=*/        [{Returns the name of the FastMathFlagsAttr attribute
                         for the operation}],
      /*returnType=*/  "StringRef",
      /*methodName=*/  "getFastMathAttrName",
      /*args=*/        (ins),
      /*methodBody=*/  [{}],
      /*defaultImpl=*/ [{
        return "fastmath";
      }]
      >

  ];
}

#endif // COMPLEX_OPS_INTERFACES
