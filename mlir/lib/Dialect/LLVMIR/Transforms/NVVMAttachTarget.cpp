//===- NVVMAttachTarget.cpp - Attach an NVVM target -----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/AttachTarget.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVM/NVVM/Target.h"
#include "llvm/Support/Regex.h"

namespace mlir {
namespace NVVM {
#define GEN_PASS_DEF_NVVMATTACHTARGET
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace NVVM
} // namespace mlir

using namespace mlir;
using namespace mlir::NVVM;

namespace {
struct NVVMAttachTarget
    : public NVVM::impl::NVVMAttachTargetBase<NVVMAttachTarget> {
  using Base::Base;

  DictionaryAttr getFlags(OpBuilder &builder) const;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registerNVVMTarget(registry);
  }
};
} // namespace

DictionaryAttr NVVMAttachTarget::getFlags(OpBuilder &builder) const {
  UnitAttr unitAttr = builder.getUnitAttr();
  SmallVector<NamedAttribute, 2> flags;
  auto addFlag = [&](StringRef flag) {
    flags.push_back(builder.getNamedAttr(flag, unitAttr));
  };
  if (fastFlag)
    addFlag("fast");
  if (ftzFlag)
    addFlag("ftz");
  if (flags.size())
    return builder.getDictionaryAttr(flags);
  return nullptr;
}

void NVVMAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  ArrayRef<std::string> libs(linkLibs);
  SmallVector<StringRef> filesToLink(libs.begin(), libs.end());
  auto target = builder.getAttr<NVVMTargetAttr>(
      optLevel, triple, chip, features, getFlags(builder),
      filesToLink.size() ? builder.getStrArrayAttr(filesToLink) : nullptr);
  llvm::Regex matcher(moduleMatcher);
  for (Region &region : getOperation()->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module : block.getOps<gpu::GPUModuleOp>()) {
        // Check if the name of the module matches.
        if (moduleMatcher.empty() || matcher.match(module.getName())) {
          // Create the target array.
          SmallVector<Attribute> targets;
          if (std::optional<ArrayAttr> attrs = module.getTargets())
            targets.append(attrs->getValue().begin(), attrs->getValue().end());
          targets.push_back(target);
          // Remove any duplicate targets.
          targets.erase(std::unique(targets.begin(), targets.end()),
                        targets.end());
          // Update the target attray.
          module.setTargetsAttr(builder.getArrayAttr(targets));
        }
      }
}

std::unique_ptr<Pass> NVVM::createNVVMAttachTarget() {
  return std::make_unique<NVVMAttachTarget>();
}

std::unique_ptr<Pass>
NVVM::createNVVMAttachTarget(const NVVMAttachTargetOptions &options) {
  return std::make_unique<NVVMAttachTarget>(options);
}
