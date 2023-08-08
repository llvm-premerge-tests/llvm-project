//===-- Passes.td - LLVM pass definition file --------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def LLVMLegalizeForExport : Pass<"llvm-legalize-for-export"> {
  let summary = "Legalize LLVM dialect to be convertible to LLVM IR";
  let constructor = "::mlir::LLVM::createLegalizeForExportPass()";
  let dependentDialects = ["LLVM::LLVMDialect"];
}

def LLVMRequestCWrappers
    : Pass<"llvm-request-c-wrappers", "::mlir::func::FuncOp"> {
  let summary = "Request C wrapper emission for all functions";
  let description = [{
    Annotate every builtin function in the module with the LLVM dialect
    attribute that instructs the conversion to LLVM to emit the C wrapper for
    the function. This pass is expected to be applied immediately before the
    conversion of builtin functions to LLVM to avoid the attribute being
    dropped by other passes.
  }];
  let constructor = "::mlir::LLVM::createRequestCWrappersPass()";
}

def LLVMTypeConsistency
    : Pass<"llvm-type-consistency", "::mlir::LLVM::LLVMFuncOp"> {
  let summary = "Rewrites to improve type consistency";
  let description = [{
    Set of rewrites to improve the coherency of types within an LLVM dialect
    program. This will adjust operations operating on pointers so they interpret
    their associated pointee type as consistently as possible.
  }];
  let constructor = "::mlir::LLVM::createTypeConsistencyPass()";

  let options = [
    Option<"maxVectorSplitSize", "max-vector-split-size", "unsigned",
           /*default=*/"512",
           "Maximum size in bits of a vector value in a load or store operation"
           " operating on multiple elements that should still be split">,
  ];
}

def NVVMOptimizeForTarget : Pass<"llvm-optimize-for-nvvm-target"> {
  let summary = "Optimize NVVM IR";
  let constructor = "::mlir::NVVM::createOptimizeForTargetPass()";
}

def DIScopeForLLVMFuncOp : Pass<"ensure-debug-info-scope-on-llvm-func", "LLVM::LLVMFuncOp"> {
  let summary = "Materialize LLVM debug info subprogram attribute on every LLVMFuncOp";
  let description = [{
    Having a debug info subprogram attribute on a function is required for
    emitting line tables from MLIR FileLocCol locations.

    This is not intended to be a proper replacement for frontends to emit
    complete debug informations, however it is a convenient way to get line
    tables for debugging purposes. This allow to step trough in a debugger
    line-by-line or get a backtrace with line numbers.
  }];

  let constructor = "mlir::LLVM::createDIScopeForLLVMFuncOpPass()";
}

def NVVMAttachTarget: Pass<"nvvm-attach-target", ""> {
  let summary = "Attaches an NVVM target attribute to a GPU Module.";
  let description = [{
    This pass searches for all GPU Modules in the immediate regions and attaches
    an NVVM target if the module matches the name specified by the `module` argument.

    Example:
    ```
    // File: in.mlir:
    gpu.module @nvvm_module_1 {...}
    gpu.module @nvvm_module_2 {...}
    gpu.module @rocdl_module_1 {...}
    // mlir-opt --nvvm-attach-target="module=nvvm.* chip=sm_90" in.mlir
    gpu.module @nvvm_module_1 [#nvvm.target<chip = "sm_90">] {...}
    gpu.module @nvvm_module_2 [#nvvm.target<chip = "sm_90">] {...}
    gpu.module @rocdl_module_1 {...}
    ```
  }];
  let options = [
    Option<"moduleMatcher", "module", "std::string",
           /*default=*/ [{""}],
           "Regex used to identify the modules to attach the target to.">,
    Option<"triple", "triple", "std::string",
           /*default=*/ "\"nvptx64-nvidia-cuda\"",
           "Target triple.">,
    Option<"chip", "chip", "std::string",
           /*default=*/"\"sm_50\"",
           "Target chip.">,
    Option<"features", "features", "std::string",
           /*default=*/"\"+ptx60\"",
           "Target features.">,
    Option<"optLevel", "O", "unsigned",
           /*default=*/"2",
           "Optimization level.">,
    Option<"fastFlag", "fast", "bool",
           /*default=*/"false",
           "Enable fast math mode.">,
    Option<"ftzFlag", "ftz", "bool",
           /*default=*/"false",
           "Enable flush to zero for denormals.">,
    ListOption<"linkLibs", "l", "std::string",
           "Extra bitcode libraries paths to link to.">,
  ];
  let constructor = "mlir::NVVM::createNVVMAttachTarget()";
}

def ROCDLAttachTarget: Pass<"rocdl-attach-target", ""> {
  let summary = "Attaches a ROCDL target attribute to a GPU Module.";
  let description = [{
    This pass searches for all GPU Modules in the immediate regions and attaches
    a ROCDL target if the module matches the name specified by the `module` argument.

    Example:
    ```
    // File: in.mlir:
    gpu.module @nvvm_module_1 {...}
    gpu.module @nvvm_module_2 {...}
    gpu.module @rocdl_module_1 {...}
    // mlir-opt --nvvm-attach-target="module=rocdl.* chip=gfx90a" in.mlir
    gpu.module @nvvm_module_1 {...}
    gpu.module @nvvm_module_2 {...}
    gpu.module @rocdl_module_1 [#rocdl.target<chip = "gfx90a">] {...}
    ```
  }];
  let options = [
    Option<"moduleMatcher", "module", "std::string",
           /*default=*/ [{""}],
           "Regex used to identify the modules to attach the target to.">,
    Option<"triple", "triple", "std::string",
           /*default=*/ "\"amdgcn-amd-amdhsa\"",
           "Target triple.">,
    Option<"chip", "chip", "std::string",
           /*default=*/"\"gfx900\"",
           "Target chip.">,
    Option<"features", "features", "std::string",
           /*default=*/"\"\"",
           "Target features.">,
    Option<"abiVersion", "abi", "std::string",
           /*default=*/"\"400\"",
           "Optimization level.">,
    Option<"optLevel", "O", "unsigned",
           /*default=*/"2",
           "Optimization level.">,
    Option<"wave64Flag", "wave64", "bool",
           /*default=*/"true",
           "Use Wave64 mode.">,
    Option<"fastFlag", "fast", "bool",
           /*default=*/"false",
           "Enable fast relaxed math opt.">,
    Option<"dazFlag", "daz", "bool",
           /*default=*/"false",
           "Enable denormals are zero opt.">,
    Option<"finiteOnlyFlag", "finite-only", "bool",
           /*default=*/"false",
           "Enable finite only opt.">,
    Option<"unsafeMathFlag", "unsafe-math", "bool",
           /*default=*/"false",
           "Enable unsafe math opt.">,
    Option<"correctSqrtFlag", "correct-sqrt", "bool",
           /*default=*/"true",
           "Enable correct rounded sqrt.">,
    ListOption<"linkLibs", "l", "std::string",
           "Extra bitcode libraries paths to link to.">,
  ];
  let constructor = "mlir::ROCDL::createROCDLAttachTarget()";
}

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_PASSES
