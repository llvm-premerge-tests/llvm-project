! RUN: not %flang -E %s 2>&1 | FileCheck %s
! Check that we diagnose wrong arguments passed to function-like macros,
! this time indirectly using another object-like macro.
subroutine foo()
  implicit none

#define presub sub
#define presub0 sub0
#define sub(x, y) foo2(x, y)
#define sub0() foo2(1, 2)
   call presub()
   call presub(1)
   call presub(1, 2, 3)
   call presub0(1, 2)
end subroutine foo

!CHECK: error: macro 'sub' requires 2 arguments, but only 1 given
!CHECK: error: macro 'sub' requires 2 arguments, but only 1 given
!CHECK: error: macro 'sub' passed 3 arguments, but takes just 2
!CHECK: error: macro 'sub0' passed 2 arguments, but takes just 0
