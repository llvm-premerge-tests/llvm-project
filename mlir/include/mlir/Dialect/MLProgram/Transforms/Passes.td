//===-- Passes.td - pass definition file -------------------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES
#define MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES

include "mlir/Pass/PassBase.td"

def MLProgramPipelineGlobals : Pass<"mlprogram-pipeline-globals", "ModuleOp"> {
  let summary = "Pipeline `ml_program` global operations for read and store";
  let constructor = "mlir::ml_program::createMLProgramPipelineGlobalsPass()";
}

#endif // MLIR_DIALECT_MLPROGRAM_TRANSFORMS_PASSES
