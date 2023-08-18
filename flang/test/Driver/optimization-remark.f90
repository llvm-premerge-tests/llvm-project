! This file tests the -Rpass family of flags (-Rpass, -Rpass-missed
! and -Rpass-analysis)
! loop-delete isn't enabled at O0 so we use at least O1

! Check that we can override -Rpass= with -Rno-pass.
! RUN: %flang_fc1 %s -O1 -Rpass -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS
! RUN: %flang_fc1 %s -O1 -Rpass -Rno-pass -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NO-REMARKS

! Check -Rno-pass, -Rno-pass-analysis, -Rno-pass-missed nothing emitted
! RUN: %flang %s -O1 -Rno-pass 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-NO-REMARKS
! RUN: %flang %s -O1 -Rno-pass-missed 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-NO-REMARKS
! RUN: %flang %s -O1 -Rno-pass-analysis 2>&1 | FileCheck %s --allow-empty --check-prefix=CHECK-NO-REMARKS

! Check full -Rpass message is emitted
! RUN: %flang %s -O1 -Rpass 2>&1 | FileCheck %s

! Check full -Rpass-missed message is emitted
! RUN: %flang %s -O1 -Rpass-missed 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS-MISSED

! Check full -Rpass-analysis message is emitted
! RUN: %flang %s -O1 -Rpass-analysis 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS-ANALYSIS

! CHECK: remark: Loop deleted because it is invariant
! CHECK-REMARKS-MISSED: remark: loop not vectorized
! CHECK-REMARKS-ANALYSIS: remark: loop not vectorized: instruction cannot be vectorized
! CHECK-REMARKS: remark:
! CHECK-NO-REMARKS-NOT: remark:

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
