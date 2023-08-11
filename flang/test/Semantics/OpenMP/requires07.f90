! RUN: %python %S/../test_errors.py %s %flang -fopenmp
! OpenMP Version 5.0
! 2.4 Requires directive
! Target-related clauses in 'requires' directives must come strictly before any
! device constructs, such as target parallel regions.

subroutine f
  !$omp target parallel
  !$omp end target parallel
end subroutine f

subroutine g
  !ERROR: 'DYNAMIC_ALLOCATORS' REQUIRES clause found lexically after device construct
  !$omp requires dynamic_allocators
  !ERROR: 'REVERSE_OFFLOAD' REQUIRES clause found lexically after device construct
  !$omp requires reverse_offload
  !ERROR: 'UNIFIED_ADDRESS' REQUIRES clause found lexically after device construct
  !$omp requires unified_address
  !ERROR: 'UNIFIED_SHARED_MEMORY' REQUIRES clause found lexically after device construct
  !$omp requires unified_shared_memory
end subroutine g
