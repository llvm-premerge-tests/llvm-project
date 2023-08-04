! RUN: %flang_fc1 -fsyntax-only -module-dir %S -fopenacc %s | cat %S/acc_mod.mod | FileCheck %s

! UNSUPPORTED: system-windows

module acc_mod
  real :: data_create(100)
  !$acc declare create(data_create)

  real :: data_copyin(10)
  !$acc declare copyin(data_copyin)

  real :: data_device_resident(20)
  !$acc declare device_resident(data_device_resident)

  integer :: data_link(50)
  !$acc declare link(data_link)
end module

! CHECK: !mod$ v1 sum:{{.*}}
! CHECK-LABEL: module acc_mod
! CHECK: real(4)::data_create(1_8:100_8)
! CHECK: !$acc declare create(data_create)
! CHECK: real(4)::data_copyin(1_8:10_8)
! CHECK: !$acc declare copyin(data_copyin)
! CHECK: real(4)::data_device_resident(1_8:20_8)
! CHECK: !$acc declare device_resident(data_device_resident)
! CHECK: integer(4)::data_link(1_8:50_8)
! CHECK: !$acc declare link(data_link)
! CHECK: end
