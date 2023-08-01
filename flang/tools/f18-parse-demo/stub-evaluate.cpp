//===-- tools/f18/stub-evaluate.cpp ---------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// The parse tree has slots in which pointers to the results of semantic
// analysis may be placed.  When using the parser without the semantics
// libraries, as here, we need to stub out the dependences on the external
// deleters and copiers, which will never actually be called.

namespace Fortran::evaluate {
struct GenericExprWrapper {
  static void Deleter(GenericExprWrapper *);
  static GenericExprWrapper *Copier(GenericExprWrapper *);
};
void GenericExprWrapper::Deleter(GenericExprWrapper *) {}
GenericExprWrapper *GenericExprWrapper::Copier(GenericExprWrapper *) {
  return nullptr;
}
struct GenericAssignmentWrapper {
  static void Deleter(GenericAssignmentWrapper *);
  static GenericAssignmentWrapper *Copier(GenericAssignmentWrapper *);
};
void GenericAssignmentWrapper::Deleter(GenericAssignmentWrapper *) {}
GenericAssignmentWrapper *GenericAssignmentWrapper::Copier(
    GenericAssignmentWrapper *) {
  return nullptr;
}
struct ProcedureRef {
  static void Deleter(ProcedureRef *);
  static ProcedureRef *Copier(ProcedureRef *);
};
void ProcedureRef::Deleter(ProcedureRef *) {}
ProcedureRef *ProcedureRef::Copier(ProcedureRef *) { return nullptr; }
} // namespace Fortran::evaluate
