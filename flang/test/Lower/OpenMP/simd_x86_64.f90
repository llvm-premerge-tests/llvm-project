! Tests for 2.9.3.1 Simd and target dependent defult alignment for x86

! REQUIRES: x86-registered-target
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-fir -fopenmp -target-cpu x86-64  %s -o - | FileCheck --check-prefixes=DEFAULT %s
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-fir -fopenmp -target-cpu x86-64 -target-feature +avx %s -o - | FileCheck --check-prefixes=AVX %s
! RUN: %flang_fc1 -triple x86_64-unknown-linux-gnu -emit-fir -fopenmp -target-cpu x86-64  -target-feature +avx512f  %s -o - | FileCheck --check-prefixes=AVX512F %s

!DEFAULT: func.func @_QPsimdloop_aligned_cptr(%[[ARG_A:.*]]: !fir.ref
!DEFAULT-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr
!DEFAULT-SAME: {__address:i64}>> {fir.bindc_name = "a"}) {
!AVX: func.func @_QPsimdloop_aligned_cptr(%[[ARG_A:.*]]: !fir.ref
!AVX-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr
!AVX-SAME: {__address:i64}>> {fir.bindc_name = "a"}) {
!AVX512F: func.func @_QPsimdloop_aligned_cptr(%[[ARG_A:.*]]: !fir.ref
!AVX512F-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr
!AVX512F-SAME: {__address:i64}>> {fir.bindc_name = "a"}) {
subroutine simdloop_aligned_cptr(A)
  use iso_c_binding
  integer :: i
  type (c_ptr) :: A
!DEFAULT: omp.simdloop aligned(%[[ARG_A]] : !fir.ref
!DEFAULT-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
!DEFAULT-SAME: -> 128 : i64)
!AVX: omp.simdloop aligned(%[[ARG_A]] : !fir.ref
!AVX-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
!AVX-SAME: -> 256 : i64)
!AVX512F: omp.simdloop aligned(%[[ARG_A]] : !fir.ref
!AVX512F-SAME: <!fir.type<_QM__fortran_builtinsT__builtin_c_ptr{__address:i64}>>
!AVX512F-SAME: -> 512 : i64)
  !$OMP SIMD ALIGNED(A)
  do i = 1, 10
    call c_test_call(A)
  end do
  !$OMP END SIMD
end subroutine

