//===- DataFlowFramework.cpp - A generic framework for data-flow analysis -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/DataFlowFramework.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "dataflow"
#if LLVM_ENABLE_ABI_BREAKING_CHECKS
#define DATAFLOW_DEBUG(X) LLVM_DEBUG(X)
#else
#define DATAFLOW_DEBUG(X)
#endif // LLVM_ENABLE_ABI_BREAKING_CHECKS

using namespace mlir;

//===----------------------------------------------------------------------===//
// GenericProgramPoint
//===----------------------------------------------------------------------===//

GenericProgramPoint::~GenericProgramPoint() = default;

//===----------------------------------------------------------------------===//
// ProgramPoint
//===----------------------------------------------------------------------===//

void ProgramPoint::print(raw_ostream &os) const {
  if (isNull()) {
    os << "<NULL POINT>";
    return;
  }
  if (auto *programPoint = dyn_cast<GenericProgramPoint *>())
    return programPoint->print(os);
  if (auto *op = dyn_cast<Operation *>())
    return op->print(os);
  if (auto value = dyn_cast<Value>())
    return value.print(os);
  return get<Block *>()->print(os);
}

Location ProgramPoint::getLoc() const {
  if (auto *programPoint = dyn_cast<GenericProgramPoint *>())
    return programPoint->getLoc();
  if (auto *op = dyn_cast<Operation *>())
    return op->getLoc();
  if (auto value = dyn_cast<Value>())
    return value.getLoc();
  return get<Block *>()->getParent()->getLoc();
}

//===----------------------------------------------------------------------===//
// AbstractState and AbstractElement
//===----------------------------------------------------------------------===//

AbstractState::~AbstractState() = default;
AbstractElement::~AbstractElement() = default;

void AbstractElement::addDependency(DataFlowAnalysis *analysis,
                                            ProgramPoint point) {
  auto inserted = dependents.insert({point, analysis});
  (void)inserted;
  DATAFLOW_DEBUG({
    if (inserted) {
      llvm::dbgs() << "Adding dependency from " << debugName << " of "
                   << this->point << " to " << analysis->debugName << " on "
                   << point << "\n";
    }
  });
}

void AbstractElement::propagateUpdate() {
  DATAFLOW_DEBUG(llvm::dbgs() << "Propagating update to " << debugName << " of "
                              << point << "\nValue: " << *get() << "\n");
  for (auto &item : dependents)
    solver.enqueue(item);
  onUpdate();
}

//===----------------------------------------------------------------------===//
// DataFlowSolver
//===----------------------------------------------------------------------===//

LogicalResult DataFlowSolver::initializeAndRun(Operation *top) {
  // Initialize the analyses.
  for (DataFlowAnalysis &analysis : llvm::make_pointee_range(childAnalyses)) {
    DATAFLOW_DEBUG(llvm::dbgs()
                   << "Priming analysis: " << analysis.debugName << "\n");
    if (failed(analysis.initialize(top)))
      return failure();
  }

  // Run the analysis until fixpoint.
  ProgramPoint point;
  DataFlowAnalysis *analysis;

  do {
    // Exhaust the worklist.
    while (!worklist.empty()) {
      std::tie(point, analysis) = worklist.front();
      worklist.pop();

      DATAFLOW_DEBUG(llvm::dbgs() << "Invoking '" << analysis->debugName
                                  << "' on: " << point << "\n");
      if (failed(analysis->visit(point)))
        return failure();
    }

    // Iterate until all states are in some initialized state and the worklist
    // is exhausted.
  } while (!worklist.empty());

  return success();
}

void DataFlowSolver::getStaticProvidersFor(
    TypeID stateID, ProgramPoint point,
    SmallVectorImpl<DataFlowAnalysis *> &staticProviders) const {
  for (DataFlowAnalysis &analysis : llvm::make_pointee_range(childAnalyses))
    if (analysis.staticallyProvides(stateID, point))
      staticProviders.push_back(&analysis);
}

//===----------------------------------------------------------------------===//
// DataFlowAnalysis
//===----------------------------------------------------------------------===//

DataFlowAnalysis::~DataFlowAnalysis() = default;

DataFlowAnalysis::DataFlowAnalysis(DataFlowSolver &solver) : solver(solver) {}
