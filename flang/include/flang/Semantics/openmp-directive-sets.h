//===-- include/flang/Semantics/openmp-directive-sets.h ---------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_
#define FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_

#include "flang/Common/enum-set.h"
#include "llvm/Frontend/OpenMP/OMPConstants.h"

using OmpDirectiveSet = Fortran::common::EnumSet<llvm::omp::Directive,
    llvm::omp::Directive_enumSize>;

namespace llvm::omp {
// Directive sets for single directives:
// - top<Directive>Set: The directive appears alone or as the first in a
//   combined construct.
// - all<Directive>Set: All standalone or combined uses of the directive.
extern OmpDirectiveSet topParallelSet;
extern OmpDirectiveSet allParallelSet;
extern OmpDirectiveSet topDoSet;
extern OmpDirectiveSet allDoSet;
extern OmpDirectiveSet topTaskloopSet;
extern OmpDirectiveSet allTaskloopSet;
extern OmpDirectiveSet topTargetSet;
extern OmpDirectiveSet allTargetSet;
extern OmpDirectiveSet topSimdSet;
extern OmpDirectiveSet allSimdSet;
extern OmpDirectiveSet topTeamsSet;
extern OmpDirectiveSet allTeamsSet;
extern OmpDirectiveSet topDistributeSet;
extern OmpDirectiveSet allDistributeSet;

// Directive sets for groups of multiple directives.
extern OmpDirectiveSet allDoSimdSet;
extern OmpDirectiveSet workShareSet;
extern OmpDirectiveSet taskGeneratingSet;
extern OmpDirectiveSet nonPartialVarSet;

// Directive sets for allowed/not allowed directives to nest inside another
// particular directive.
extern OmpDirectiveSet nestedOrderedErrSet;
extern OmpDirectiveSet nestedWorkshareErrSet;
extern OmpDirectiveSet nestedMasterErrSet;
extern OmpDirectiveSet nestedBarrierErrSet;
extern OmpDirectiveSet nestedTeamsAllowedSet;
extern OmpDirectiveSet nestedOrderedParallelErrSet;
extern OmpDirectiveSet nestedOrderedDoAllowedSet;
extern OmpDirectiveSet nestedCancelTaskgroupAllowedSet;
extern OmpDirectiveSet nestedCancelSectionsAllowedSet;
extern OmpDirectiveSet nestedCancelDoAllowedSet;
extern OmpDirectiveSet nestedCancelParallelAllowedSet;
extern OmpDirectiveSet nestedReduceWorkshareAllowedSet;
} // namespace llvm::omp

#endif // FORTRAN_SEMANTICS_OPENMP_DIRECTIVE_SETS_H_