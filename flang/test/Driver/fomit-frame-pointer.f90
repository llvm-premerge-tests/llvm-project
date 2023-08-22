! Test no error emitted with 'fomit-frame-pointer' flag.

! Test opt_record flags get generated for fc1
! RUN: %flang -fomit-frame-pointer %s

program forttest
    implicit none
    integer :: n

    n = 1

end program forttest
