//===- SerializeROCDLTarget.cpp ---------------------------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/InitAllDialects.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "mlir/Target/LLVMIR/Dialect/Builtin/BuiltinToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/GPU/GPUToLLVMIRTranslation.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMToLLVMIRTranslation.h"

#include "llvm/IRReader/IRReader.h"
#include "llvm/Support/MemoryBufferRef.h"
#include "llvm/Support/TargetSelect.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/TargetParser/Host.h"

#include "gmock/gmock.h"

using namespace mlir;

#if MLIR_ROCM_CONVERSIONS_ENABLED == 1
TEST(MLIRTargetLLVM, SerializeROCDLModule) {
  std::string moduleStr = R"mlir(
  gpu.module @kernels {
    llvm.func @kernel(%arg0: f32) attributes {gpu.kernel, rocdl.kernel} {
      llvm.return
    }
  }
  )mlir";

  DialectRegistry registry;
  registerBuiltinDialectTranslation(registry);
  registerLLVMDialectTranslation(registry);
  registerGPUDialectTranslation(registry);
  registerROCDLTarget(registry);
  MLIRContext context(registry);

  OwningOpRef<ModuleOp> module =
      parseSourceString<ModuleOp>(moduleStr, &context);
  ASSERT_TRUE(!!module);

  // Create a ROCDL target.
  ROCDL::ROCDLTargetAttr target = ROCDL::ROCDLTargetAttr::get(&context);

  // Serialize the module.
  auto serializer = dyn_cast<gpu::TargetAttrInterface>(target);
  ASSERT_TRUE(!!serializer);
  for (auto gpuModule : (*module).getBody()->getOps<gpu::GPUModuleOp>()) {
    std::optional<SmallVector<char, 0>> object =
        serializer.serializeToObject(gpuModule, {});
    // Check that the serializer was successful.
    ASSERT_TRUE(object != std::nullopt);
    ASSERT_TRUE(object->size() > 0);
  }
}
#endif
