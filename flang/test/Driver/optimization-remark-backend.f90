! This file tests backend passes emitted by the -Rpass family of flags
! loop-delete isn't enabled at O0 so we use at least O1

! DEFINE: %{output} = -S -o /dev/null 2>&1

! RUN: %flang %s -O1 -Rpass-missed %{output} 2>&1 | FileCheck %s --check-prefix=MISSED
! RUN: %flang %s -O1 -Rpass-analysis %{output} 2>&1 | FileCheck %s --check-prefix=ANALYSIS

! MISSED-DAG:   remark: {{[0-9]+}} virtual registers copies {{.*}} total copies cost generated in function [-Rpass-missed=regalloc]
! ANALYSIS-DAG: remark: BasicBlock:
! ANALYSIS-DAG: [-Rpass-analysis=asm-printer]

program forttest
    implicit none
    integer :: n

    do n = 1,2
        print *, ""
    end do

end program forttest
