//===-- lib/Semantics/openmp-directive-sets.cpp ---------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "flang/Semantics/openmp-directive-sets.h"

//===----------------------------------------------------------------------===//
// Single directives
//===----------------------------------------------------------------------===//

OmpDirectiveSet llvm::omp::topParallelSet{
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
};

OmpDirectiveSet llvm::omp::allParallelSet{
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_target_parallel,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
};

OmpDirectiveSet llvm::omp::topDoSet{
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
};

OmpDirectiveSet llvm::omp::allDoSet{
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_do,
    Directive::OMPD_do_simd,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
};

OmpDirectiveSet llvm::omp::topTaskloopSet{
    Directive::OMPD_taskloop,
    Directive::OMPD_taskloop_simd,
};

OmpDirectiveSet llvm::omp::allTaskloopSet = llvm::omp::topTaskloopSet;

OmpDirectiveSet llvm::omp::topTargetSet{
    Directive::OMPD_target,
    Directive::OMPD_target_parallel,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_simd,
    Directive::OMPD_target_teams,
    Directive::OMPD_target_teams_distribute,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_simd,
};

OmpDirectiveSet llvm::omp::allTargetSet = llvm::omp::topTargetSet;

OmpDirectiveSet llvm::omp::topSimdSet{
    Directive::OMPD_simd,
};

OmpDirectiveSet llvm::omp::allSimdSet{
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_do_simd,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_simd,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_simd,
    Directive::OMPD_taskloop_simd,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_simd,
};

OmpDirectiveSet llvm::omp::topTeamsSet{
    Directive::OMPD_teams,
    Directive::OMPD_teams_distribute,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_simd,
};

OmpDirectiveSet llvm::omp::allTeamsSet{
    llvm::omp::OMPD_target_teams,
    llvm::omp::OMPD_target_teams_distribute,
    llvm::omp::OMPD_target_teams_distribute_parallel_do,
    llvm::omp::OMPD_target_teams_distribute_parallel_do_simd,
    llvm::omp::OMPD_target_teams_distribute_simd,
    llvm::omp::OMPD_teams,
    llvm::omp::OMPD_teams_distribute,
    llvm::omp::OMPD_teams_distribute_parallel_do,
    llvm::omp::OMPD_teams_distribute_parallel_do_simd,
    llvm::omp::OMPD_teams_distribute_simd,
};

OmpDirectiveSet llvm::omp::topDistributeSet{
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
};

OmpDirectiveSet llvm::omp::allDistributeSet{
    llvm::omp::OMPD_distribute,
    llvm::omp::OMPD_distribute_parallel_do,
    llvm::omp::OMPD_distribute_parallel_do_simd,
    llvm::omp::OMPD_distribute_simd,
    llvm::omp::OMPD_target_teams_distribute,
    llvm::omp::OMPD_target_teams_distribute_parallel_do,
    llvm::omp::OMPD_target_teams_distribute_parallel_do_simd,
    llvm::omp::OMPD_target_teams_distribute_simd,
    llvm::omp::OMPD_teams_distribute,
    llvm::omp::OMPD_teams_distribute_parallel_do,
    llvm::omp::OMPD_teams_distribute_parallel_do_simd,
    llvm::omp::OMPD_teams_distribute_simd,
};

//===----------------------------------------------------------------------===//
// Groups of multiple directives
//===----------------------------------------------------------------------===//

OmpDirectiveSet llvm::omp::allDoSimdSet =
    llvm::omp::allDoSet & llvm::omp::allSimdSet;

OmpDirectiveSet llvm::omp::workShareSet{
    OmpDirectiveSet{
        Directive::OMPD_workshare,
        Directive::OMPD_parallel_workshare,
        Directive::OMPD_parallel_sections,
        Directive::OMPD_sections,
        Directive::OMPD_single,
    } | allDoSet,
};

OmpDirectiveSet llvm::omp::taskGeneratingSet{
    OmpDirectiveSet{
        Directive::OMPD_task,
    } | allTaskloopSet,
};

OmpDirectiveSet llvm::omp::nonPartialVarSet{
    Directive::OMPD_allocate,
    Directive::OMPD_allocators,
    Directive::OMPD_threadprivate,
    Directive::OMPD_declare_target,
};

OmpDirectiveSet llvm::omp::loopConstructSet{
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_simd,
    Directive::OMPD_distribute,
    Directive::OMPD_do_simd,
    Directive::OMPD_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_do,
    Directive::OMPD_simd,
    Directive::OMPD_target_parallel_do_simd,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_simd,
    Directive::OMPD_target_teams_distribute_parallel_do_simd,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_target_teams_distribute_simd,
    Directive::OMPD_target_teams_distribute,
    Directive::OMPD_taskloop_simd,
    Directive::OMPD_taskloop,
    Directive::OMPD_teams_distribute_parallel_do_simd,
    Directive::OMPD_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_simd,
    Directive::OMPD_teams_distribute,
    Directive::OMPD_tile,
    Directive::OMPD_unroll,
};

OmpDirectiveSet llvm::omp::blockConstructSet{
    Directive::OMPD_master,
    Directive::OMPD_ordered,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_parallel,
    Directive::OMPD_single,
    Directive::OMPD_target_data,
    Directive::OMPD_target_parallel,
    Directive::OMPD_target_teams,
    Directive::OMPD_target,
    Directive::OMPD_task,
    Directive::OMPD_taskgroup,
    Directive::OMPD_teams,
    Directive::OMPD_workshare,
};

//===----------------------------------------------------------------------===//
// Allowed/Not allowed nested directives
//===----------------------------------------------------------------------===//

OmpDirectiveSet llvm::omp::nestedOrderedErrSet{
    Directive::OMPD_critical,
    Directive::OMPD_ordered,
    Directive::OMPD_atomic,
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

OmpDirectiveSet llvm::omp::nestedWorkshareErrSet{
    OmpDirectiveSet{
        Directive::OMPD_task,
        Directive::OMPD_taskloop,
        Directive::OMPD_critical,
        Directive::OMPD_ordered,
        Directive::OMPD_atomic,
        Directive::OMPD_master,
    } | workShareSet,
};

OmpDirectiveSet llvm::omp::nestedMasterErrSet{
    OmpDirectiveSet{
        Directive::OMPD_atomic,
    } | taskGeneratingSet |
        workShareSet,
};

OmpDirectiveSet llvm::omp::nestedBarrierErrSet{
    OmpDirectiveSet{
        Directive::OMPD_critical,
        Directive::OMPD_ordered,
        Directive::OMPD_atomic,
        Directive::OMPD_master,
    } | taskGeneratingSet |
        workShareSet,
};

OmpDirectiveSet llvm::omp::nestedTeamsAllowedSet{
    Directive::OMPD_parallel,
    Directive::OMPD_parallel_do,
    Directive::OMPD_parallel_do_simd,
    Directive::OMPD_parallel_master,
    Directive::OMPD_parallel_master_taskloop,
    Directive::OMPD_parallel_master_taskloop_simd,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
    Directive::OMPD_distribute,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_distribute_parallel_do_simd,
    Directive::OMPD_distribute_simd,
};

OmpDirectiveSet llvm::omp::nestedOrderedParallelErrSet{
    Directive::OMPD_parallel,
    Directive::OMPD_target_parallel,
    Directive::OMPD_parallel_sections,
    Directive::OMPD_parallel_workshare,
};

OmpDirectiveSet llvm::omp::nestedOrderedDoAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
};

OmpDirectiveSet llvm::omp::nestedCancelTaskgroupAllowedSet{
    Directive::OMPD_task,
    Directive::OMPD_taskloop,
};

OmpDirectiveSet llvm::omp::nestedCancelSectionsAllowedSet{
    Directive::OMPD_sections,
    Directive::OMPD_parallel_sections,
};

OmpDirectiveSet llvm::omp::nestedCancelDoAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_distribute_parallel_do,
    Directive::OMPD_parallel_do,
    Directive::OMPD_target_parallel_do,
    Directive::OMPD_target_teams_distribute_parallel_do,
    Directive::OMPD_teams_distribute_parallel_do,
};

OmpDirectiveSet llvm::omp::nestedCancelParallelAllowedSet{
    Directive::OMPD_parallel,
    Directive::OMPD_target_parallel,
};

OmpDirectiveSet llvm::omp::nestedReduceWorkshareAllowedSet{
    Directive::OMPD_do,
    Directive::OMPD_sections,
    Directive::OMPD_do_simd,
};
