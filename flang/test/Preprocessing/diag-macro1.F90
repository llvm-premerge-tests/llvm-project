! RUN: not %flang -E %s 2>&1 | FileCheck %s
! Check that we diagnose wrong arguments passed to function-like macros.
subroutine foo()
  implicit none

#define sub(x, y) foo2(x, y)
#define sub0() foo2(1, 2)
   call sub()
   call sub(1)
   call sub(1, 2, 3)
   call sub0(1, 2)
end subroutine foo

!CHECK: error: macro 'sub' requires 2 arguments, but only 1 given
!CHECK: error: macro 'sub' requires 2 arguments, but only 1 given
!CHECK: error: macro 'sub' passed 3 arguments, but takes just 2
!CHECK: error: macro 'sub0' passed 2 arguments, but takes just 0
