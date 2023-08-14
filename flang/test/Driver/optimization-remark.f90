! This file tests the -Rpass family of flags (-Rpass, -Rpass-missed
! and -Rpass-analysis)
! loop-delete isn't enabled at O0 so we use at least O1

! Check that we can override -Rpass= with -Rno-pass.
! RUN: %flang_fc1 %s -O1 -Rpass -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS
! RUN: %flang_fc1 %s -O1 -Rpass -Rno-pass -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NO-REMARKS

! Check "unknown remark option" warning
! RUN: %flang %s -O1 -R 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS-WARN

! Check "unknown remark option" warning with suggestion
! RUN: %flang %s -O1 -Rpas 2>&1 | FileCheck %s --check-prefix=CHECK-WARN-SUGGEST

! Check full -Rpass message is emitted
! RUN: %flang %s -O1 -Rpass 2>&1 | FileCheck %s

! Check full -Rpass-missed message is emitted
! RUN: %flang %s -O1 -Rpass-missed 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS-MISSED

! Check full -Rpass-analysis message is emitted
! RUN: %flang %s -O1 -Rpass-analysis 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS-ANALYSIS

! CHECK: optimization-remark.f90:43:5: remark: Loop deleted because it is invariant [-Rpass=loop-delete]
! CHECK-REMARKS-MISSED: optimization-remark.f90:38:5: remark: loop not vectorized [-Rpass-missed=loop-vectorize]
! CHECK-REMARKS-ANALYSIS: optimization-remark.f90:40:9: remark: loop not vectorized: call instruction cannot be vectorized [-Rpass-analysis=loop-vectorize]
! CHECK-REMARKS: remark:
! CHECK-NO-REMARKS-NOT: remark:

! CHECK-REMARKS-WARN: warning: unknown remark option '-R' [-Wunknown-warning-option]
! CHECK-WARN-SUGGEST: warning: unknown remark option '-Rpas'; did you mean '-Rpass'? [-Wunknown-warning-option]

program forttest
    implicit none
    real, dimension(1:50) :: aR1
    integer :: n

    do n = 1,50
        aR1(n) = n * 1.34
        print *, "hello"
    end do

    do n = 1,50
        aR1(n) = n * 1.34
    end do

end program forttest