! This file tests the -Rpass family of flags (-Rpass, -Rpass-missed
! and -Rpass-analysis)
! loop-delete isn't enabled at O0 so we use at least O1

! Check that we can override -Rpass= with -Rno-pass.
! RUN: %flang_fc1 %s -O1 -Rpass -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=REMARKS
! RUN: %flang_fc1 %s -O1 -Rpass -Rno-pass -emit-llvm -o - 2>&1 | FileCheck %s --check-prefix=NO-REMARKS

! Check -Rno-pass, -Rno-pass-analysis, -Rno-pass-missed nothing emitted
! RUN: %flang %s -O1 -Rno-pass -c 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-REMARKS
! RUN: %flang %s -O1 -Rno-pass-missed -c 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-REMARKS
! RUN: %flang %s -O1 -Rno-pass-analysis -c 2>&1 | FileCheck %s --allow-empty --check-prefix=NO-REMARKS

! Check valid -Rpass regex
! RUN: %flang %s -O1 -Rpass=loop -c 2>&1 | FileCheck %s --check-prefix=PASS-REGEX

! Check valid -Rpass-missed regex
! RUN: %flang %s -O1 -Rpass-missed=loop -c 2>&1 | FileCheck %s --check-prefix=MISSED-REGEX

! Check valid -Rpass-analysis regex
! RUN: %flang %s -O1 -Rpass-analysis=loop -c 2>&1 | FileCheck %s --check-prefix=ANALYSIS-REGEX

! Check full -Rpass message is emitted
! RUN: %flang %s -O1 -Rpass -c 2>&1 | FileCheck %s --check-prefix=PASS

! Check full -Rpass-missed message is emitted
! RUN: %flang %s -O1 -Rpass-missed -c 2>&1 | FileCheck %s --check-prefix=MISSED

! Check full -Rpass-analysis message is emitted
! RUN: %flang %s -O1 -Rpass-analysis -c 2>&1 | FileCheck %s --check-prefix=ANALYSIS


! PASS:               remark: hoisting load
! PASS:               remark: Loop deleted because it is invariant

! PASS-REGEX-NOT:     remark: hoisting load
! PASS-REGEX:         remark: Loop deleted because it is invariant

! MISSED:             remark: failed to hoist load with loop-invariant address because load is conditionally executed
! MISSED:             remark: loop not vectorized
! MISSED-NOT:         remark: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
!                     Unknown data dependence.

! MISSED-REGEX-NOT:   remark: failed to hoist load with loop-invariant address because load is conditionally executed
! MISSED-REGEX:       remark: loop not vectorized

! ANALYSIS-REGEX:     remark: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
!                     Unknown data dependence.
! ANALYSIS-REGEX-NOT: remark: {{.*}}: IR instruction count changed from {{[0-9]+}} to {{[0-9]+}}; Delta: {{[0-9]+}}

! ANALYSIS:           remark: loop not vectorized: unsafe dependent memory operations in loop. Use #pragma loop distribute(enable) to allow loop distribution to attempt to isolate the offending operations into a separate loop
!                     Unknown data dependence.
! ANALYSIS:           remark: {{.*}}: IR instruction count changed from {{[0-9]+}} to {{[0-9]+}}; Delta: {{[0-9]+}}
! ANALYSIS-NOT:       remark: failed to hoist load with loop-invariant address because load is conditionally executed

! REMARKS: remark:
! NO-REMARKS-NOT: remark:

subroutine swap_real(a1, a2)
   implicit none

   real, dimension(1:2) :: aR1
   integer :: i, n
   real, intent(inout) :: a1(:), a2(:)
   real :: a

!  Swap
   do i = 1, min(size(a1), size(a2))
      a = a1(i)
      a1(i) = a2(i)
      a2(i) = a
   enddo

! Do a random loop to generate a successful loop-delete pass
    do n = 1,2
        aR1(n) = n * 1.34
    end do

end subroutine swap_real
