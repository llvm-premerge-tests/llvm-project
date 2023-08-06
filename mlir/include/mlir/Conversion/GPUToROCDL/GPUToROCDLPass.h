//===- GPUToROCDLPass.h - Convert GPU kernel to ROCDL dialect ---*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#ifndef MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
#define MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_

#include "mlir/Conversion/GPUToROCDL/Runtimes.h"
#include "mlir/Conversion/LLVMCommon/LoweringOptions.h"
#include <memory>

namespace llvm {
class StringRef;
} // namespace llvm

namespace mlir {
class LLVMTypeConverter;
class ConversionTarget;
class RewritePatternSet;
class Type;

template <typename OpT>
class OperationPass;

namespace gpu {
class GPUModuleOp;
class MMAMatrixType;
} // namespace gpu

#define GEN_PASS_DECL_CONVERTGPUOPSTOROCDLOPS
#include "mlir/Conversion/Passes.h.inc"

namespace amd {
/// Constant representing 32 workitems in a workgroup.
const unsigned kWaveFrontSize32 = 32;

/// Constant representing 64 workitems in a workgroup.
const unsigned kWaveFrontSize64 = 64;

/// Wavefront sizes that are supported by the GPU to ROCDL lowerings.
const unsigned kWMMASupportedWaveFrontSizes[] = {kWaveFrontSize32,
                                                 kWaveFrontSize64};

/// Return the LLVM Type corresponding to the MMAMatrixType.
Type convertWMMAToROCDLLLVMType(gpu::MMAMatrixType matrixType);
} // namespace amd

/// Collect a set of patterns to convert from the GPU dialect to ROCDL.
/// If `runtime` is Unknown, gpu.printf will not be lowered. The resulting
/// pattern set should be run over a gpu.module op. `chipset` is the chip we are
/// targeting. `warpSize` is the warp size to use when generating WMMA
/// intrinsics. `opSelect` is used in the lowering of f16 versions of WMMA ops
/// involving `C` operand. If `opSelect` is true upper half of the general
/// purpose 32-bit registers is used for storing the values; If false the lower
/// half is used.
void populateGpuToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                          RewritePatternSet &patterns,
                                          gpu::amd::Runtime runtime,
                                          llvm::StringRef chipset = "gfx900",
                                          bool opSelec = false,
                                          unsigned warpSize = 32);

/// Configure target to convert from the GPU dialect to ROCDL.
void configureGpuToROCDLConversionLegality(ConversionTarget &target);

/// Creates a pass that lowers GPU dialect operations to ROCDL counterparts. The
/// index bitwidth used for the lowering of the device side index computations
/// is configurable. AMD gpus have a configurable warp size; valid choices are
/// 32 and 64. We choose 32 as the default size. `opSelect` is used in the
/// lowering of f16 versions of WMMA ops involving `C` operand. If `opSelect` is
/// true upper half of the general purpose 32-bit registers is used for storing
/// the values; If false the lower half is used.
std::unique_ptr<OperationPass<gpu::GPUModuleOp>>
createLowerGpuOpsToROCDLOpsPass(
    const std::string &chipset = "gfx900",
    unsigned indexBitwidth = kDeriveIndexBitwidthFromDataLayout,
    bool useBarePtrCallConv = false,
    gpu::amd::Runtime runtime = gpu::amd::Runtime::Unknown,
    bool opSelect = false, unsigned warpSize = 32);

/// Collect a set of patterns to convert WMMA ops from GPU dialect to ROCDL.
/// `chipset` is the target chip for which the IR is being generated.
/// `warpSize` is the warp size to use when generating WMMA intrinsics.
void populateGpuWMMAToROCDLConversionPatterns(LLVMTypeConverter &converter,
                                              RewritePatternSet &patterns,
                                              llvm::StringRef chipset,
                                              bool opSelect, unsigned warpSize);

} // namespace mlir

#endif // MLIR_CONVERSION_GPUTOROCDL_GPUTOROCDLPASS_H_
