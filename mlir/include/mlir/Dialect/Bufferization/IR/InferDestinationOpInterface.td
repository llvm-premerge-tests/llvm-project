//===-- InferDestinationOpInterface.td - Infer destination -*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef INFER_DESTINATION_OP_INTERFACE
#define INFER_DESTINATION_OP_INTERFACE

include "mlir/IR/OpBase.td"

def InferDestinationOpInterface : OpInterface<"InferDestinationOpInterface"> {
  let description = [{
    This interface can be implemented by ops that read data from an "input"
    tensor and store the result into an "output"/"init" tensor (i.e., the
    "destination" tensor). It drives the empty tensor elimination pass.

    The `getOrBuildDestination` interface method returns the destination tensor
    (or a slice thereof). Assuming that the op does not bufferize to a memory
    read on the destination tensor (or the aforementioned slice), if the source
    originates from a "tensor.empty", that "tensor.empty" can be replaced with
    the result of `getOrBuildDestination`. This can reduce the number of
    allocations and memcpys during bufferization: instead of computing data in
    a temporary buffer (the result of "tensor.empty") and then copying it into
    the destination buffer, the result will be computed in the destination
    buffer directly.

    Example:
    ```
    %0 = tensor.empty() : tensor<5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
    %2 = tensor.insert_slice %1 into %dst [%offset][5][1]
        : tensor<5xf32> into tensor<?xf32>
    ```

    "tensor.insert_slice" transfers %1 into a slice of %dst.
    `getOrBuildDestination` should the slice. `getNeededValues` should return
    %dst and %offset, because these values are needed to build the destination.

    After empty tensor elimination:
    ```
    %0 = tensor.extract_slice %dst [%offset][5][1]
        : tensor<?xf32> to tensor<5xf32>
    %1 = linalg.fill ins(%cst : f32) outs(%0 : tensor<5xf32>) -> tensor<5xf32>
    %2 = tensor.insert_slice %1 into %dst [%offset][5][1]
        : tensor<5xf32> into tensor<?xf32>
    ```

    This IR will bufferize without an allocation and without a memcpy.
  }];
  let cppNamespace = "::mlir::bufferization";
  let methods = [
      InterfaceMethod<
        /*desc=*/[{
          Build or return the destination tensor (or a slice thereof) into which
          the given operand (or an element-wise computation thereof) will be
          stored. Only values returned by `getNeededValues` may be used to build
          the destination. This interface method will be called only on tensor
          operands for which `isAnchor` returns "true".
        }],
        /*retType=*/"::mlir::Value",
        /*methodName=*/"getOrBuildDestination",
        /*args=*/(ins "::mlir::OpBuilder &":$builder,
                      "::mlir::Location":$loc,
                      "::mlir::OpOperand &":$operand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          llvm_unreachable("getOrBuildDestination not implemented");
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return the SSA values that are needed to build the destination tensor.
          This interface method will be called only on tensor operands for which
          `isAnchor` returns "true"
        }],
        /*retType=*/"::llvm::SmallVector<::mlir::Value>",
        /*methodName=*/"getNeededValues",
        /*args=*/(ins "::mlir::OpOperand &":$operand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          llvm_unreachable("getNeededValues not implemented");
        }]
      >,
      InterfaceMethod<
        /*desc=*/[{
          Return "true" if the given operand is an "input" tensor on which
          empty tensor elimination can be applied.
        }],
        /*retType=*/"bool",
        /*methodName=*/"isAnchor",
        /*args=*/(ins "::mlir::OpOperand &":$operand),
        /*methodBody=*/"",
        /*defaultImplementation=*/[{
          llvm_unreachable("isAnchor not implemented");
        }]
      >,
  ];
}

#endif // INFER_DESTINATION_OP_INTERFACE
