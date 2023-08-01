#ifndef LLVM_TRANSFORMS_FUNCTIONANNOTATOR_H
#define LLVM_TRANSFORMS_FUNCTIONANNOTATOR_H

#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"

namespace llvm {
class FunctionAnnotator : public PassInfoMixin<FunctionAnnotator> {
public:
  PreservedAnalyses run(Module &M, ModuleAnalysisManager &AM);
  static cl::opt<std::string> CSVFilePath;
};

} // namespace llvm

#endif // LLVM_TRANSFORMS_FUNCTIONANNOTATOR_H
