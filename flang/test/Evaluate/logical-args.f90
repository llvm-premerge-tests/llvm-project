! Test that actual logical arguments convert to the right kind when it is non-default
! RUN: %flang_fc1 -fdebug-unparse %s 2>&1 | FileCheck %s
! RUN: %flang_fc1 -fdebug-unparse -fdefault-integer-8 %s 2>&1 | FileCheck %s --check-prefixes CHECK-8

program main
  integer :: x(10), y
  logical(kind=1) :: l(2)
  ! CHECK: CALL foo(.true._4)
  ! CHECK-8: CALL foo(logical(.true._4,kind=8))
  call foo(1 < 2)
  ! CHECK: CALL fooa(x>y)
  ! CHECK-8: CALL fooa(logical(x>y,kind=8))
  call fooa(x > y)

  ! CHECK: CALL bar(.true._4)
  ! CHECK-8: CALL bar(logical(.true._4,kind=8))
  call bar(1 < 2)

  ! Make sure we don't try to convert when the argument comes from an intrinsic
  ! call with a non-default kind logical result
  ! CHECK: CALL baz(any(l))
  ! CHECK-8: CALL baz(any(l))
  call baz(any(l))

  contains
    subroutine foo(l)
      logical :: l
    end subroutine foo

    subroutine fooa(l)
      logical :: l(10)
    end subroutine fooa
end program main
