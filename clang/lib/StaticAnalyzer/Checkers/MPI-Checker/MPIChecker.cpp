//===-- MPIChecker.cpp - Checker Entry Point Class --------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the main class of MPI-Checker which serves as an entry
/// point. It is created once for each translation unit analysed.
/// The checker defines path-sensitive checks, to verify correct usage of the
/// MPI API.
///
//===----------------------------------------------------------------------===//

#include "MPIChecker.h"
#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/DynamicExtent.h"

namespace clang {
namespace ento {
namespace mpi {

void MPIChecker::checkDoubleNonblocking(const CallEvent &PreCallEvent,
                                        CheckerContext &Ctx) const {
  if (!FuncClassifier->isNonBlockingType(PreCallEvent.getCalleeIdentifier())) {
    return;
  }
  const MemRegion *const MR =
      PreCallEvent.getArgSVal(PreCallEvent.getNumArgs() - 1).getAsRegion();
  if (!MR)
    return;
  const ElementRegion *const ER = dyn_cast<ElementRegion>(MR);

  // The region must be typed, in order to reason about it.
  if (!isa<TypedRegion>(MR) || (ER && !isa<TypedRegion>(ER->getSuperRegion())))
    return;

  ProgramStateRef State = Ctx.getState();
  const Request *const Req = State->get<RequestMap>(MR);

  // double nonblocking detected
  if (Req && Req->CurrentState == Request::State::Nonblocking) {
    ExplodedNode *ErrorNode = Ctx.generateNonFatalErrorNode();
    BReporter.reportDoubleNonblocking(PreCallEvent, *Req, MR, ErrorNode,
                                      Ctx.getBugReporter());
    Ctx.addTransition(ErrorNode->getState(), ErrorNode);
  }
  // no error
  else {
    State = State->set<RequestMap>(MR, Request::State::Nonblocking);
    Ctx.addTransition(State);
  }
}

void MPIChecker::checkUnmatchedWaits(const CallEvent &PreCallEvent,
                                     CheckerContext &Ctx) const {
  if (!FuncClassifier->isWaitType(PreCallEvent.getCalleeIdentifier()))
    return;
  const MemRegion *const MR = topRegionUsedByWait(PreCallEvent);
  if (!MR)
    return;
  const ElementRegion *const ER = dyn_cast<ElementRegion>(MR);

  // The region must be typed, in order to reason about it.
  if (!isa<TypedRegion>(MR) || (ER && !isa<TypedRegion>(ER->getSuperRegion())))
    return;

  llvm::SmallVector<const MemRegion *, 2> ReqRegions;
  allRegionsUsedByWait(ReqRegions, MR, PreCallEvent, Ctx);
  if (ReqRegions.empty())
    return;

  ProgramStateRef State = Ctx.getState();
  static CheckerProgramPointTag Tag("MPI-Checker", "UnmatchedWait");
  ExplodedNode *ErrorNode{nullptr};

  // Check all request regions used by the wait function.
  for (const auto &ReqRegion : ReqRegions) {
    const Request *const Req = State->get<RequestMap>(ReqRegion);
    State = State->set<RequestMap>(ReqRegion, Request::State::Wait);
    if (!Req) {
      if (!ErrorNode) {
        ErrorNode = Ctx.generateNonFatalErrorNode(State, &Tag);
        State = ErrorNode->getState();
      }
      // A wait has no matching nonblocking call.
      BReporter.reportUnmatchedWait(PreCallEvent, ReqRegion, ErrorNode,
                                    Ctx.getBugReporter());
    }
  }

  if (!ErrorNode) {
    Ctx.addTransition(State);
  } else {
    Ctx.addTransition(State, ErrorNode);
  }
}

void MPIChecker::checkMissingWaits(SymbolReaper &SymReaper,
                                   CheckerContext &Ctx) const {
  ProgramStateRef State = Ctx.getState();
  const auto &Requests = State->get<RequestMap>();
  if (Requests.isEmpty())
    return;

  static CheckerProgramPointTag Tag("MPI-Checker", "MissingWait");
  ExplodedNode *ErrorNode{nullptr};

  auto ReqMap = State->get<RequestMap>();
  for (const auto &Req : ReqMap) {
    if (!SymReaper.isLiveRegion(Req.first)) {
      if (Req.second.CurrentState == Request::State::Nonblocking) {

        if (!ErrorNode) {
          ErrorNode = Ctx.generateNonFatalErrorNode(State, &Tag);
          State = ErrorNode->getState();
        }
        BReporter.reportMissingWait(Req.second, Req.first, ErrorNode,
                                    Ctx.getBugReporter());
      }
      State = State->remove<RequestMap>(Req.first);
    }
  }

  // Transition to update the state regarding removed requests.
  if (!ErrorNode) {
    Ctx.addTransition(State);
  } else {
    Ctx.addTransition(State, ErrorNode);
  }
}

const MemRegion *MPIChecker::topRegionUsedByWait(const CallEvent &CE) const {

  if (FuncClassifier->isMPI_Wait(CE.getCalleeIdentifier())) {
    return CE.getArgSVal(0).getAsRegion();
  } else if (FuncClassifier->isMPI_Waitall(CE.getCalleeIdentifier())) {
    return CE.getArgSVal(1).getAsRegion();
  } else {
    return (const MemRegion *)nullptr;
  }
}

static std::optional<std::pair<NonLoc, llvm::APSInt>>
getRequestRegionOffsetAndCount(const MemRegion *const MR, const CallEvent &CE) {
  if (CE.getNumArgs() < 2)
    return std::nullopt;

  ProgramStateRef State = CE.getState();
  SValBuilder &SVB = State->getStateManager().getSValBuilder();
  ASTContext &ASTCtx = SVB.getContext();

  QualType RequestTy = CE.getArgExpr(1)->getType()->getPointeeType();
  auto RequestRegionCount =
      getDynamicElementCountWithOffset(State, CE.getArgSVal(1), RequestTy)
          .getAs<nonloc::ConcreteInt>();
  if (!RequestRegionCount)
    return std::nullopt;

  CharUnits TypeSizeInChars = ASTCtx.getTypeSizeInChars(RequestTy);

  // MPI_Request as a handle does not have to be of non-zero size.
  int64_t TypeSizeInBits =
      (TypeSizeInChars.isZero() ? 1 : TypeSizeInChars.getQuantity()) *
      ASTCtx.getCharWidth();

  RegionOffset RequestRegionOffset = MR->getAsOffset();
  if (RequestRegionOffset.hasSymbolicOffset())
    return std::nullopt;

  return std::make_pair(
      SVB.makeArrayIndex(RequestRegionOffset.getOffset() / TypeSizeInBits),
      RequestRegionCount->getValue());
}

void MPIChecker::allRegionsUsedByWait(
    llvm::SmallVector<const MemRegion *, 2> &ReqRegions,
    const MemRegion *const MR, const CallEvent &CE, CheckerContext &Ctx) const {

  MemRegionManager &RegionManager = MR->getMemRegionManager();

  if (FuncClassifier->isMPI_Waitall(CE.getCalleeIdentifier())) {
    const SubRegion *SuperRegion{nullptr};
    if (const ElementRegion *const ER = MR->getAs<ElementRegion>()) {
      SuperRegion = cast<SubRegion>(ER->getSuperRegion());
    }

    // A single request is passed to MPI_Waitall.
    if (!SuperRegion) {
      ReqRegions.push_back(MR);
      return;
    }

    auto RequestRegionOffsetAndCount = getRequestRegionOffsetAndCount(MR, CE);
    if (!RequestRegionOffsetAndCount)
      return;

    auto [RegionOffset, RegionCount] = *RequestRegionOffsetAndCount;

    QualType MPIReqTy = CE.getArgExpr(1)->getType()->getPointeeType();
    SValBuilder &SVB = Ctx.getSValBuilder();

    auto RequestedCountSVal = CE.getArgSVal(0).getAs<nonloc::ConcreteInt>();
    if (!RequestedCountSVal)
      return;

    const llvm::APSInt &RequestedCount = RequestedCountSVal->getValue();
    // TODO: i >= RegionCount is an OOB UB, we could report it here but a better
    // approach is adding this constraint as a summary into generic checker like
    // StdCLibraryFunctions
    for (size_t i = 0; i < RegionCount && i < RequestedCount; ++i) {
      auto RegionIndex =
          SVB.evalBinOp(Ctx.getState(), BO_Add, SVB.makeArrayIndex(i),
                        RegionOffset, SVB.getArrayIndexType())
              .getAs<NonLoc>();
      if (RegionIndex) {
        const ElementRegion *const RequestRegion =
            RegionManager.getElementRegion(MPIReqTy, *RegionIndex, SuperRegion,
                                           Ctx.getASTContext());
        ReqRegions.push_back(RequestRegion);
      }
    }
  } else if (FuncClassifier->isMPI_Wait(CE.getCalleeIdentifier())) {
    ReqRegions.push_back(MR);
  }
}

} // end of namespace: mpi
} // end of namespace: ento
} // end of namespace: clang

// Registers the checker for static analysis.
void clang::ento::registerMPIChecker(CheckerManager &MGR) {
  MGR.registerChecker<clang::ento::mpi::MPIChecker>();
}

bool clang::ento::shouldRegisterMPIChecker(const CheckerManager &mgr) {
  return true;
}
