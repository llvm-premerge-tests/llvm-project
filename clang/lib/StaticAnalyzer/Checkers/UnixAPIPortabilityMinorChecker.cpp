//= UnixAPIPortabilityMinorChecker.cpp --------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This files defines the UnixAPIPortabilityMinorChecker, which is a collection
// of checks for minor portability issues in calls to various UNIX/Posix
// functions.
//
//===----------------------------------------------------------------------===//

#include "clang/StaticAnalyzer/Checkers/BuiltinCheckerRegistration.h"
#include "clang/StaticAnalyzer/Core/BugReporter/BugType.h"
#include "clang/StaticAnalyzer/Core/Checker.h"
#include "clang/StaticAnalyzer/Core/CheckerManager.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallDescription.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CallEvent.h"
#include "clang/StaticAnalyzer/Core/PathSensitive/CheckerContext.h"

using namespace clang;
using namespace ento;

namespace {
class UnixAPIPortabilityMinorChecker : public Checker<check::PreCall> {
public:
  UnixAPIPortabilityMinorChecker();

  void checkPreCall(const CallEvent &Call, CheckerContext &C) const;

private:
  std::unique_ptr<BugType> PrintfPointerConversionSpecifierNULLBugType;
  void CheckPrintfFamilyPointerConversionSpecifierNULLGeneric(
      const CallEvent &Call, CheckerContext &C,
      unsigned int data_args_index) const;
  void CheckPrintfFamilyPointerConversionSpecifierNULLPrintf(
      const CallEvent &Call, CheckerContext &C) const;
  void CheckPrintfFamilyPointerConversionSpecifierNULLFprintf(
      const CallEvent &Call, CheckerContext &C) const;
  void CheckPrintfFamilyPointerConversionSpecifierNULLSprintf(
      const CallEvent &Call, CheckerContext &C) const;
  void CheckPrintfFamilyPointerConversionSpecifierNULLSnprintf(
      const CallEvent &Call, CheckerContext &C) const;
  void ReportPrintfPointerConversionSpecifierNULL(CheckerContext &C,
                                                  ProgramStateRef nullState,
                                                  const Expr *arg) const;
  using CheckFn = void (UnixAPIPortabilityMinorChecker::*)(
      const CallEvent &Call, CheckerContext &C) const;
  const CallDescriptionMap<CheckFn> PrintfFamilyFunctions = {
      {{{"printf"}},
       &UnixAPIPortabilityMinorChecker::
           CheckPrintfFamilyPointerConversionSpecifierNULLPrintf},
      {{{"fprintf"}},
       &UnixAPIPortabilityMinorChecker::
           CheckPrintfFamilyPointerConversionSpecifierNULLFprintf},
      {{{"sprintf"}},
       &UnixAPIPortabilityMinorChecker::
           CheckPrintfFamilyPointerConversionSpecifierNULLSprintf},
      {{{"snprintf"}},
       &UnixAPIPortabilityMinorChecker::
           CheckPrintfFamilyPointerConversionSpecifierNULLSnprintf}};
};
} // end anonymous namespace

UnixAPIPortabilityMinorChecker::UnixAPIPortabilityMinorChecker() {
  PrintfPointerConversionSpecifierNULLBugType.reset(
      new BugType(this,
                  "Passing a null pointer to the pointer conversion "
                  "specifier of ",
                  categories::UnixAPI));
}

//===----------------------------------------------------------------------===//
// printf family of functions with null pointer passed to pointer
// conversion specifier
//===----------------------------------------------------------------------===//

// Generates an error report, indicating that the result of passing a null
// pointer to pointer conversion specifier of printf family of functions is
// implementation defined.
void UnixAPIPortabilityMinorChecker::ReportPrintfPointerConversionSpecifierNULL(
    CheckerContext &C, ProgramStateRef nullState, const Expr *arg) const {
  ExplodedNode *N =
      C.generateNonFatalErrorNode(nullState ? nullState : C.getState());
  if (!N)
    return;
  auto report = std::make_unique<PathSensitiveBugReport>(
      *PrintfPointerConversionSpecifierNULLBugType,
      "The result of passing a null pointer to the pointer conversion "
      "specifier of the printf family of functions is implementation defined",
      N);
  report->addRange(arg->getSourceRange());
  bugreporter::trackExpressionValue(N, arg, *report);
  C.emitReport(std::move(report));
}

// Checks data arguments of printf family of functions for a null pointer,
// assuming it is passed to a pointer conversion specifier (%p), i.e. without
// checking the format string.
void UnixAPIPortabilityMinorChecker::
    CheckPrintfFamilyPointerConversionSpecifierNULLGeneric(
        const CallEvent &Call, CheckerContext &C,
        unsigned int data_args_index) const {
  ProgramStateRef state = C.getState();
  ConstraintManager &CM = state->getConstraintManager();

  for (unsigned int i = data_args_index; i < Call.getNumArgs(); i++) {
    const Expr *arg = Call.getArgExpr(i);
    if (!arg)
      continue;

    const QualType type = arg->getType();
    if (!type->isPointerType() && !type->isNullPtrType())
      continue;

    SVal argVal = Call.getArgSVal(i);
    if (argVal.isUnknownOrUndef())
      continue;

    auto argDefinedVal = argVal.getAs<DefinedSVal>();

    ProgramStateRef notNullState, nullState;
    std::tie(notNullState, nullState) =
        CM.assumeDual(C.getState(), *argDefinedVal);
    if (!notNullState && nullState) {
      ReportPrintfPointerConversionSpecifierNULL(C, nullState, arg);
      return;
    }
  }
}

void UnixAPIPortabilityMinorChecker::
    CheckPrintfFamilyPointerConversionSpecifierNULLPrintf(
        const CallEvent &Call, CheckerContext &C) const {
  CheckPrintfFamilyPointerConversionSpecifierNULLGeneric(Call, C, 1);
}

void UnixAPIPortabilityMinorChecker::
    CheckPrintfFamilyPointerConversionSpecifierNULLFprintf(
        const CallEvent &Call, CheckerContext &C) const {
  CheckPrintfFamilyPointerConversionSpecifierNULLGeneric(Call, C, 2);
}

void UnixAPIPortabilityMinorChecker::
    CheckPrintfFamilyPointerConversionSpecifierNULLSprintf(
        const CallEvent &Call, CheckerContext &C) const {
  CheckPrintfFamilyPointerConversionSpecifierNULLGeneric(Call, C, 2);
}

void UnixAPIPortabilityMinorChecker::
    CheckPrintfFamilyPointerConversionSpecifierNULLSnprintf(
        const CallEvent &Call, CheckerContext &C) const {
  CheckPrintfFamilyPointerConversionSpecifierNULLGeneric(Call, C, 3);
}

void UnixAPIPortabilityMinorChecker::checkPreCall(const CallEvent &Call,
                                                  CheckerContext &C) const {
  if (!Call.isGlobalCFunction())
    return;

  if (const auto *Checker = PrintfFamilyFunctions.lookup(Call))
    (this->**Checker)(Call, C);
}

//===----------------------------------------------------------------------===//
// Checker registration.
//===----------------------------------------------------------------------===//

void ento::registerUnixAPIPortabilityMinorChecker(CheckerManager &mgr) {
  mgr.registerChecker<UnixAPIPortabilityMinorChecker>();
}
bool ento::shouldRegisterUnixAPIPortabilityMinorChecker(
    const CheckerManager &mgr) {
  return true;
}
