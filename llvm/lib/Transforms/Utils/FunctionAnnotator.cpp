#include "llvm/Transforms/Utils/FunctionAnnotator.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/LineIterator.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Testing/Annotations/Annotations.h"

using namespace llvm;

PreservedAnalyses FunctionAnnotator::run(Module &M, ModuleAnalysisManager &AM) {

  static cl::opt<std::string> OptLevelAttributeName(
      "-opt-level-attribute-name", cl::init(""), cl::Hidden,
      cl::desc("Optimization attribute name"));

  static cl::opt<std::string> CSVFilePath(
      "-csv-file-path", cl::Hidden, cl::Required,
      cl::desc("CSV file containing function names and optimization level as "
               "attribute"));

  auto BufferOrError = MemoryBuffer::getFileOrSTDIN(CSVFilePath);
  if (!BufferOrError) {
    report_fatal_error("Cannot open CSV File");
  }

  StringRef Buffer = BufferOrError.get()->getBuffer();
  auto memoryBuffer = MemoryBuffer::getMemBuffer(Buffer);
  line_iterator itr(*memoryBuffer, false);
  for (Function &F : M) {
    if (F.isDeclaration()) {
      continue;
    }

    while (!itr.is_at_end()) {
      if (itr.operator*().split(',').second != "") {
        if (itr.operator*().split(',').first == F.getName()) {
          F.addFnAttr(OptLevelAttributeName, itr.operator*().split(',').second);
          break;
        }
      }
      ++itr;
    }
  }

  return PreservedAnalyses::all();
}