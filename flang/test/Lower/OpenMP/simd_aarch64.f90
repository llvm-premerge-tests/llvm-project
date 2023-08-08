! Tests for 2.9.3.1 Simd and target dependent defult alignment for AArch64
! The default alignment for AARch is 0 so we do not emit aligned clause

! REQUIRES: aarch64-registered-target
! RUN: %flang_fc1 -triple aarch64-unknown-linux-gnu -emit-fir -fopenmp %s -o - | FileCheck  %s

subroutine simdloop_aligned_cptr(A)
  use iso_c_binding
  integer :: i
  type (c_ptr) :: A
!CHECK: omp.simdloop
!CHECK-NOT: aligned(
!CHECK-SAME: for (%[[IT:.*]]) : i32 = (%[[LB:.*]]) to (%[[UB:.*]]) inclusive step (%[[INC:.*]]) {
  !$OMP SIMD ALIGNED(A)
  do i = 1, 10
    call c_test_call(A)
  end do
  !$OMP END SIMD
end subroutine

