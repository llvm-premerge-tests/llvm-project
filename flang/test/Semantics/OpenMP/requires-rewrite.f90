! RUN: %flang_fc1 -fopenmp -fdebug-dump-parse-tree %s 2>&1 | FileCheck %s
! Ensure that requires atomic_default_mem_order is used to update atomic
! operations with no explicit memory order set.
program requires
  implicit none
  !$omp requires atomic_default_mem_order(seq_cst)
  integer :: i, j, k

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomic
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> SeqCst
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic relaxed
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomic
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> SeqCst
  !$omp atomic
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicUpdate
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> SeqCst
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic relaxed update
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicUpdate
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> SeqCst
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic update relaxed
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicUpdate
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> SeqCst
  !$omp atomic update
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicCapture
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> SeqCst
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic relaxed capture
  i = j
  j = k
  !$omp end atomic

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicCapture
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> SeqCst
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic capture relaxed
  i = j
  j = k
  !$omp end atomic

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicCapture
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> SeqCst
  !$omp atomic capture
  i = j
  j = k
  !$omp end atomic

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicWrite
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> SeqCst
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic relaxed write
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicWrite
  ! CHECK-NOT: OmpMemoryOrderClause -> OmpClause -> SeqCst
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> Relaxed
  !$omp atomic write relaxed
  i = j

  ! CHECK-LABEL: OpenMPAtomicConstruct -> OmpAtomicWrite
  ! CHECK: OmpMemoryOrderClause -> OmpClause -> SeqCst
  !$omp atomic write
  i = j
end program requires
