! RUN: bbc %s -o - | tco | FileCheck %s

! CHECK-LABEL: define void @_QQmain()
program p
  ! CHECK-DAG:  alloca { ptr, i64, i32, i8, i8, i8, i8 }, align 8
  ! CHECK-DAG:  alloca { ptr, i64, i32, i8, i8, i8, i8 }, align 8
  ! CHECK-DAG:  alloca { ptr, i64, i32, i8, i8, i8, i8 }, align 8
  ! CHECK-DAG:  alloca { ptr, i64, i32, i8, i8, i8, i8 }, align 8
  ! CHECK-DAG:  alloca x86_fp80, i64 1, align 16
  ! CHECK-DAG:  alloca fp128, i64 1, align 16
  ! CHECK-DAG:  alloca half, i64 1, align 2
  ! CHECK-DAG:  alloca bfloat, i64 1, align 2
  ! CHECK-DAG:  alloca float, i64 1, align 4
  ! CHECK-DAG:  alloca double, i64 1, align 8
  character(10) :: in = 'NaN'
  real(kind=2)  :: x2
  real(kind=3)  :: x3
  real(kind=4)  :: x4
  real(kind=8)  :: x8
  real(kind=10) :: x10
  real(kind=16) :: x16

  read(in,*) x2
  ! CHECK:      insertvalue { ptr, i64, i32, i8, i8, i8, i8 } { ptr undef, i64 ptrtoint (ptr getelementptr (half, ptr null, i32 1) to i64), i32 {{[0-9]*}}, i8 0, i8 25, i8 0, i8 0 }, ptr %{{[0-9]*}}, 0
  ! CHECK:      call i1 @_FortranAioOutputDescriptor(ptr %{{[0-9]*}}, ptr %{{[0-9]*}})
  print "(z4)", x2

  read(in,*) x3
  ! CHECK:      insertvalue { ptr, i64, i32, i8, i8, i8, i8 } { ptr undef, i64 ptrtoint (ptr getelementptr (bfloat, ptr null, i32 1) to i64), i32 {{[0-9]*}}, i8 0, i8 26, i8 0, i8 0 }, ptr %{{[0-9]*}}, 0
  ! CHECK:      call i1 @_FortranAioOutputDescriptor(ptr %{{[0-9]*}}, ptr %{{[0-9]*}})
  print "(z4)", x3

  read(in,*) x4
  ! CHECK:      call i1 @_FortranAioOutputReal32(ptr %{{[0-9]*}}, float %{{[0-9]*}})
  print "(z8)", x4

  read(in,*) x8
  ! CHECK:      call i1 @_FortranAioOutputReal64(ptr %{{[0-9]*}}, double %{{[0-9]*}})
  print "(z16)", x8

  read(in,*) x10
  ! CHECK:      insertvalue { ptr, i64, i32, i8, i8, i8, i8 } { ptr undef, i64 ptrtoint (ptr getelementptr (x86_fp80, ptr null, i32 1) to i64), i32 {{[0-9]*}}, i8 0, i8 29, i8 0, i8 0 }, ptr %{{[0-9]*}}, 0
  ! CHECK:      call i1 @_FortranAioOutputDescriptor(ptr %{{[0-9]*}}, ptr %{{[0-9]*}})
  print "(z20)", x10

  read(in,*) x16
  ! CHECK:      insertvalue { ptr, i64, i32, i8, i8, i8, i8 } { ptr undef, i64 ptrtoint (ptr getelementptr (fp128, ptr null, i32 1) to i64), i32 {{[0-9]*}}, i8 0, i8 31, i8 0, i8 0 }, ptr %{{[0-9]*}}, 0
  ! CHECK:      call i1 @_FortranAioOutputDescriptor(ptr %{{[0-9]*}}, ptr %{{[0-9]*}})
  print "(z32)", x16
end
