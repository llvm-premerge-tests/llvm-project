#ifndef LLVM_CODEGEN_EXPANDVAINTRINSICS_H
#define LLVM_CODEGEN_EXPANDVAINTRINSICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

class ExpandVAIntrinsicsPass : public PassInfoMixin<ExpandVAIntrinsicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_EXPANDVAINTRINSICS_H
