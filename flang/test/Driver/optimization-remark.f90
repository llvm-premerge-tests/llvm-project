! This file tests the -Rpass family of flags (-Rpass, -Rpass-missed
! and -Rpass-analysis)
! loop-delete isn't enabled at O0 so we use at least O1

! Check that we can override -Rpass= with -Rno-pass.
! RUN: %flang_fc1 %s -O1 -Rpass=loop-delete -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS
! RUN: %flang_fc1 %s -O1 -Rpass=loop-delete -Rno-pass -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NO-REMARKS
! RUN: %flang_fc1 %s -O1 -Rpass=loop-delete -Rno-everything -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-NO-REMARKS
! RUN: %flang_fc1 %s -O1 -Rpass=loop-delete -Rno-everything -Reverything -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS

! -Reverything implies -Rpass=.*.
! RUN: %flang_fc1 %s -O1 -Reverything -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS

! -Rpass implies -Rpass=.*
! RUN: %flang_fc1 %s -O1 -Rpass -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=CHECK-REMARKS

! Check full -Rpass message is emitted
! RUN: %flang %s -O1 -Rpass=loop-delete 2>&1 | FileCheck %s


! CHECK: remark: Loop deleted because it is invariant
! CHECK-REMARKS: remark:
! CHECK-NO-REMARKS-NOT: remark:


program forttest
    implicit none
    real, dimension(1:50) :: aR1
    integer :: n

    do n = 1,50
        aR1(n) = n * 1
    end do

end program forttest
