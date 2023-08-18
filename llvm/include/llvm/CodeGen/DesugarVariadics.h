#ifndef LLVM_CODEGEN_DESUGARVARIADICS_H
#define LLVM_CODEGEN_DESUGARVARIADICS_H

#include "llvm/IR/PassManager.h"

namespace llvm {

class Module;

class DesugarVariadicsPass : public PassInfoMixin<DesugarVariadicsPass> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
};

} // end namespace llvm

#endif // LLVM_CODEGEN_DESUGAR_VARIADICS_H
