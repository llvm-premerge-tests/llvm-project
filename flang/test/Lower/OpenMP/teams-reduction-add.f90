! RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s

! CHECK: omp.reduction.declare @[[REDUCTION_NAME:.*]] : i32 init

! CHECK: func @_QPteams_reduction_add
subroutine teams_reduction_add()
  ! CHECK: %[[I:.*]] = fir.alloca i32
  integer :: i
  i = 0

  ! CHECK: omp.teams
  ! CHECK-SAME: reduction(@[[REDUCTION_NAME]] -> %[[I]] : !fir.ref<i32>)
  !$omp teams reduction(+:i)
  ! CHECK: omp.reduction %{{.*}}, %[[I]] : i32, !fir.ref<i32>
  ! CHECK-NEXT: omp.terminator
  i = i + 1
  !$omp end teams
end subroutine teams_reduction_add
