//===-- CompilationAttrInterfaces.td - GPU compilation interfaces ---------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines interfaces for GPU target attributes & GPU object manager
// attributes.
//
//===----------------------------------------------------------------------===//

#ifndef GPU_COMPILATIONATTRINTERFACES
#define GPU_COMPILATIONATTRINTERFACES

include "mlir/IR/AttrTypeBase.td"
include "mlir/IR/OpBase.td"

//===----------------------------------------------------------------------===//
// GPU target attribute interface.
//===----------------------------------------------------------------------===//

def GPUTargetAttrInterface : AttrInterface<"TargetAttrInterface"> {
  let description = [{
    Interface for GPU target attributes. Attributes implementing this interface
    compile GPU modules into binary objects, providing an opaque interface to
    hide implementation details.
  }];
  let cppNamespace = "::mlir::gpu";
  let methods = [
    InterfaceMethod<[{
        Serializes a GPU module to a string containing a representation of the
        module.

        If serialization fails then the method should return `std::nullopt`.

        The `options` argument is meant to be used for passing additional
        options that are not in the attribute.
      }],
      "std::optional<SmallVector<char, 0>>", "serializeToObject",
      (ins "Operation*":$module, "const gpu::TargetOptions&":$options)>
  ];
}

def ImplementsTargetAttrInterface : AttrConstraint<
  CPred<"::mlir::isa<::mlir::gpu::TargetAttrInterface>($_self)">,
  "Attribute implementing the `TargetAttrInterface` interface."
>;

def GPUTargetAttr : ConfinedAttr<AnyAttr, [ImplementsTargetAttrInterface]> {
  let description = [{
    Generic target attribute implementing the `TargetAttrInterface` interface.
  }];
}

def GPUTargetArrayAttr :
  TypedArrayAttrBase<GPUTargetAttr, "Array of GPU target attributes">;

def GPUNonEmptyTargetArrayAttr :
  ConfinedAttr<GPUTargetArrayAttr, [ArrayMinCount<1>]>;

//===----------------------------------------------------------------------===//
// GPU object manager attribute interface.
//===----------------------------------------------------------------------===//

def GPUBinaryHandlerLLVMTranslationAttrInterface :
    AttrInterface<"BinaryHandlerLLVMTranslationAttrInterface"> {
  let description = [{
    Interface for GPU object manager attributes. Attributes implementing this
    interface manage the interaction between GPU objects and host IR.
  }];
  let cppNamespace = "::mlir::gpu";
  let methods = [
    InterfaceMethod<[{
        Embeds a GPU object into a host LLVM module. The operation expected by
        this method must be a GPU BinaryOp.

        All attributes implementing this interface must implement this method.
        If the method fails then it must return `failure`.
      }],
      "LogicalResult", "embedBinary",
      (ins "Operation*":$binaryOp, "llvm::IRBuilderBase&":$hostBuilder,
           "LLVM::ModuleTranslation&":$hostModuleTranslation)
    >,
    InterfaceMethod<[{
        Launches a kernel inside a binary. The first argument must be a GPU
        LaunchFuncOp, while the second one a GPU BinaryOp.

        All attributes implementing this interface must implement this method.
        If the method fails then it must return `failure`.
      }],
      "LogicalResult", "launchKernel",
      (ins "Operation*":$launchFunc, "Operation*":$binaryOp,
           "llvm::IRBuilderBase&":$hostBuilder,
           "LLVM::ModuleTranslation&":$hostModuleTranslation)
    >
  ];
}

def ImplementsBinaryHandlerLLVMTranslationAttrInterface : AttrConstraint<
  CPred<"isa<::mlir::gpu::BinaryHandlerLLVMTranslationAttrInterface>($_self)">,
  "Attribute implementing the `BinaryHandlerLLVMTranslationAttrInterface` interface."
>;

def GPUBinaryHandlerLLVMTranslationAttr :
    ConfinedAttr<AnyAttr, [ImplementsBinaryHandlerLLVMTranslationAttrInterface]> {
  let description = [{
    Generic compilation attribute implementing the `BinaryHandlerLLVMTranslationAttrInterface`
    interface.
  }];
}

#endif // GPU_COMPILATIONATTRINTERFACES
