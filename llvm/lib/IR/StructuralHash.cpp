//===-- StructuralHash.cpp - IR Hashing -------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/IR/StructuralHash.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/GlobalVariable.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"

using namespace llvm;

namespace {

// Basic hashing mechanism to detect structural change to the IR, used to verify
// pass return status consistency with actual change. Loosely copied from
// llvm/lib/Transforms/Utils/FunctionComparator.cpp

class StructuralHashImpl {
  hash_code Hash;

  template <typename T> void hash(const T &V) { Hash = hash_combine(Hash, V); }

public:
  StructuralHashImpl() : Hash(4) {}

  // A function hash is calculated by considering only the number of arguments
  // and whether a function is varargs, the order of basic blocks (given by the
  // successors of each basic block in depth first order), and the order of
  // opcodes of each instruction within each of these basic blocks. This mirrors
  // the strategy FunctionComparator::compare() uses to compare functions by
  // walking the BBs in depth first order and comparing each instruction in
  // sequence. Because this hash currently does not look at the operands, it is
  // insensitive to things such as the target of calls and the constants used in
  // the function, which makes it useful when possibly merging functions which
  // are the same modulo constants and call targets.
  //
  // Note that different users of StructuralHash will want different behavior
  // out of it (i.e., MergeFunctions will want something different from PM
  // expensive checks for pass modification status). When modifying this
  // function, most changes should be gated behind an option and enabled
  // selectively.
  void update(const Function &F, bool DetailedHash) {
    // Declarations don't affect analyses.
    if (F.isDeclaration())
      return;

    hash(0x6acaa36bef8325c5ULL); // Function header

    hash(F.isVarArg());
    hash(F.arg_size());

    SmallVector<const BasicBlock *, 8> BBs;
    SmallPtrSet<const BasicBlock *, 16> VisitedBBs;

    // Walk the blocks in the same order as
    // FunctionComparator::cmpBasicBlocks(), accumulating the hash of the
    // function "structure." (BB and opcode sequence)
    BBs.push_back(&F.getEntryBlock());
    VisitedBBs.insert(BBs[0]);
    while (!BBs.empty()) {
      const BasicBlock *BB = BBs.pop_back_val();

      // This random value acts as a block header, as otherwise the partition of
      // opcodes into BBs wouldn't affect the hash, only the order of the
      // opcodes
      hash(45798);
      for (auto &Inst : *BB) {
        hash(Inst.getOpcode());
        if (DetailedHash) {
          hash(Inst.getType());

          // Handle additional properties of specific instructions that cause
          // semantic differences in the IR.
          if (const CmpInst *ComparisonInstruction = dyn_cast<CmpInst>(&Inst))
            hash(ComparisonInstruction->getPredicate());
          if (const IntrinsicInst *InstrinsicInstruction =
                  dyn_cast<IntrinsicInst>(&Inst))
            hash(InstrinsicInstruction->getIntrinsicID());
          if (const CallInst *CallInstruction = dyn_cast<CallInst>(&Inst))
            hash(CallInstruction->getCalledFunction()->getName());

          for (unsigned I = 0; I < Inst.getNumOperands(); ++I) {
            hash(Inst.getOperand(I)->getType());
          }
        }
      }

      const Instruction *Term = BB->getTerminator();
      for (unsigned i = 0, e = Term->getNumSuccessors(); i != e; ++i) {
        if (!VisitedBBs.insert(Term->getSuccessor(i)).second)
          continue;
        BBs.push_back(Term->getSuccessor(i));
      }
    }
  }

  void update(const GlobalVariable &GV) {
    // Declarations and used/compiler.used don't affect analyses.
    // Since there are several `llvm.*` metadata, like `llvm.embedded.object`,
    // we ignore anything with the `.llvm` prefix
    if (GV.isDeclaration() || GV.getName().starts_with("llvm."))
      return;
    hash(23456); // Global header
    hash(GV.getValueType()->getTypeID());
  }

  void update(const Module &M, bool DetailedHash) {
    for (const GlobalVariable &GV : M.globals())
      update(GV);
    for (const Function &F : M)
      update(F, DetailedHash);
  }

  uint64_t getHash() const { return Hash; }
};

} // namespace

IRHash llvm::StructuralHash(const Function &F, bool DetailedHash) {
  StructuralHashImpl H;
  H.update(F, DetailedHash);
  return H.getHash();
}

IRHash llvm::StructuralHash(const Module &M, bool DetailedHash) {
  StructuralHashImpl H;
  H.update(M, DetailedHash);
  return H.getHash();
}
