! RUN: not %flang -E %s 2>&1 | FileCheck %s
! Check that we diagnose unterminated function-like macro invocations
subroutine foo()
  implicit none
#define sub(x, y) foo2(x, y)
   call sub(1, 2
   print *, "hello"
end subroutine foo

!CHECK: error: unterminated argument list invoking macro 'sub'
