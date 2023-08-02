! RUN: %python %S/../test_errors.py %s %flang_fc1 -fopenmp
! OpenMP Version 5.0
! 2.11.3 allocate Directive
! List items specified in an allocate directive that is associated
! with an allocate statement must be variables that are allocated
! by the allocate statement.

subroutine allocate()
use omp_lib
  integer, dimension(:), allocatable :: a, b, c, d, e, f

  !$omp allocate(a) allocator(omp_default_mem_alloc)
    allocate(a(1), b(2))

  !$omp allocate(c, d) allocator(omp_default_mem_alloc)
    allocate(c(3), d(4))

  !ERROR: Object 'e' in ALLOCATE directive not found in corresponding ALLOCATE statement
  !$omp allocate(e, f) allocator(omp_default_mem_alloc)
    allocate(f(6))

end subroutine allocate
