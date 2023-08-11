//===- ROCDLAttachTarget.cpp - Attach an ROCDL target
//-----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/LLVMIR/Transforms/AttachTarget.h"

#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/IR/Builders.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Target/LLVM/ROCDL/Target.h"
#include "llvm/Support/Regex.h"

namespace mlir {
namespace ROCDL {
#define GEN_PASS_DEF_ROCDLATTACHTARGET
#include "mlir/Dialect/LLVMIR/Transforms/Passes.h.inc"
} // namespace ROCDL
} // namespace mlir

using namespace mlir;
using namespace mlir::ROCDL;

namespace {
struct ROCDLAttachTarget
    : public ROCDL::impl::ROCDLAttachTargetBase<ROCDLAttachTarget> {
  using Base::Base;

  DictionaryAttr getFlags(OpBuilder &builder) const;

  void runOnOperation() override;

  void getDependentDialects(DialectRegistry &registry) const override {
    registerROCDLTarget(registry);
  }
};
} // namespace

DictionaryAttr ROCDLAttachTarget::getFlags(OpBuilder &builder) const {
  UnitAttr unitAttr = builder.getUnitAttr();
  SmallVector<NamedAttribute, 6> flags;
  auto addFlag = [&](StringRef flag) {
    flags.push_back(builder.getNamedAttr(flag, unitAttr));
  };
  if (!wave64Flag)
    addFlag("no_wave64");
  if (fastFlag)
    addFlag("fast");
  if (dazFlag)
    addFlag("daz");
  if (finiteOnlyFlag)
    addFlag("finite_only");
  if (unsafeMathFlag)
    addFlag("unsafe_math");
  if (!correctSqrtFlag)
    addFlag("unsafe_sqrt");
  if (flags.size())
    return builder.getDictionaryAttr(flags);
  return nullptr;
}

void ROCDLAttachTarget::runOnOperation() {
  OpBuilder builder(&getContext());
  ArrayRef<std::string> libs(linkLibs);
  SmallVector<StringRef> filesToLink(libs.begin(), libs.end());
  auto target = builder.getAttr<ROCDLTargetAttr>(
      optLevel, triple, chip, features, abiVersion, getFlags(builder),
      filesToLink.size() ? builder.getStrArrayAttr(filesToLink) : nullptr);
  llvm::Regex matcher(moduleMatcher);
  for (Region &region : getOperation()->getRegions())
    for (Block &block : region.getBlocks())
      for (auto module : block.getOps<gpu::GPUModuleOp>()) {
        // Check if the name of the module matches.
        if (!moduleMatcher.empty() && !matcher.match(module.getName()))
          continue;
        // Create the target array.
        SmallVector<Attribute> targets;
        if (std::optional<ArrayAttr> attrs = module.getTargets())
          targets.append(attrs->getValue().begin(), attrs->getValue().end());
        targets.push_back(target);
        // Remove any duplicate targets.
        targets.erase(std::unique(targets.begin(), targets.end()),
                      targets.end());
        // Update the target attribute array.
        module.setTargetsAttr(builder.getArrayAttr(targets));
      }
}

std::unique_ptr<Pass> ROCDL::createROCDLAttachTarget() {
  return std::make_unique<ROCDLAttachTarget>();
}

std::unique_ptr<Pass>
ROCDL::createROCDLAttachTarget(const ROCDLAttachTargetOptions &options) {
  return std::make_unique<ROCDLAttachTarget>(options);
}
