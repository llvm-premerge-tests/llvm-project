//===- AttachTarget.h - Attaches GPU targets to GPU Modules -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_DIALECT_LLVMIR_TRANSFORMS_ATTACHTARGET_H
#define MLIR_DIALECT_LLVMIR_TRANSFORMS_ATTACHTARGET_H

#include "mlir/Pass/Pass.h"
#include <memory>

namespace mlir {
namespace NVVM {
#define GEN_PASS_DECL_NVVMATTACHTARGET
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

/// Creates a pass that attaches NVVM targets to GPU Module Ops.
std::unique_ptr<Pass> createNVVMAttachTarget();

/// Creates a pass that attaches NVVM targets to GPU Module Ops using `options`.
std::unique_ptr<Pass>
createNVVMAttachTarget(const NVVMAttachTargetOptions &options);
} // namespace NVVM

namespace ROCDL {
#define GEN_PASS_DECL_ROCDLATTACHTARGET
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"

/// Creates a pass that attaches ROCDL targets to GPU Module Ops.
std::unique_ptr<Pass> createROCDLAttachTarget();

/// Creates a pass that attaches ROCDL targets to GPU Module Ops using
/// `options`.
std::unique_ptr<Pass>
createROCDLAttachTarget(const ROCDLAttachTargetOptions &options);
} // namespace ROCDL
} // namespace mlir

#endif // MLIR_DIALECT_LLVMIR_TRANSFORMS_ATTACHTARGET_H
