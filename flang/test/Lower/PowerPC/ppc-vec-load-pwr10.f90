! RUN: %flang_fc1 -target-cpu pwr10 -emit-fir %s -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang_fc1 -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="LLVMIR-P10" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_lxvp
!----------------------

! CHECK-LABEL: @vec_lxvp_test_i2_
subroutine vec_lxvp_test_i2(v1, offset, vp)
  implicit none
  integer(2) :: offset
  vector(integer(2)) :: v1
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_i2

! CHECK-LABEL: @vec_lxvp_test_i4_
subroutine vec_lxvp_test_i4(v1, offset, vp)
  implicit none
  integer(2) :: offset
  vector(integer(4)) :: v1
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_i4

! CHECK-LABEL: @vec_lxvp_test_u2_
subroutine vec_lxvp_test_u2(v1, offset, vp)
  implicit none
  integer(2) :: offset
  vector(unsigned(2)) :: v1
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.vector<8:ui16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_u2

! CHECK-LABEL: @vec_lxvp_test_u4_
subroutine vec_lxvp_test_u4(v1, offset, vp)
  implicit none
  integer(2) :: offset
  vector(unsigned(4)) :: v1
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.vector<4:ui32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_u4

! CHECK-LABEL: @vec_lxvp_test_r4_
subroutine vec_lxvp_test_r4(v1, offset, vp)
  implicit none
  integer(2) :: offset
  vector(real(4)) :: v1
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.vector<4:f32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_r4

! CHECK-LABEL: @vec_lxvp_test_r8_
subroutine vec_lxvp_test_r8(v1, offset, vp)
  implicit none
  integer(2) :: offset
  vector(real(8)) :: v1
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.vector<2:f64>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_r8

! CHECK-LABEL: @vec_lxvp_test_vp_
subroutine vec_lxvp_test_vp(v1, offset, vp)
  implicit none
  integer(2) :: offset
  __vector_pair :: v1
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.vector<256:i1>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_vp

! CHECK-LABEL: @vec_lxvp_test_i2_arr_
subroutine vec_lxvp_test_i2_arr(v1, offset, vp)
  implicit none
  integer :: offset
  vector(integer(2)) :: v1(10)
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<10x!fir.vector<8:i16>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_i2_arr

! CHECK-LABEL: @vec_lxvp_test_i4_arr_
subroutine vec_lxvp_test_i4_arr(v1, offset, vp)
  implicit none
  integer :: offset
  vector(integer(4)) :: v1(10)
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<10x!fir.vector<4:i32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_i4_arr

! CHECK-LABEL: @vec_lxvp_test_u2_arr_
subroutine vec_lxvp_test_u2_arr(v1, offset, vp)
  implicit none
  integer :: offset
  vector(unsigned(2)) :: v1(10)
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<10x!fir.vector<8:ui16>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_u2_arr

! CHECK-LABEL: @vec_lxvp_test_u4_arr_
subroutine vec_lxvp_test_u4_arr(v1, offset, vp)
  implicit none
  integer :: offset
  vector(unsigned(4)) :: v1(10)
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<10x!fir.vector<4:ui32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_u4_arr

! CHECK-LABEL: @vec_lxvp_test_r4_arr_
subroutine vec_lxvp_test_r4_arr(v1, offset, vp)
  implicit none
  integer :: offset
  vector(real(4)) :: v1(10)
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<10x!fir.vector<4:f32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_r4_arr

! CHECK-LABEL: @vec_lxvp_test_r8_arr_
subroutine vec_lxvp_test_r8_arr(v1, offset, vp)
  implicit none
  integer :: offset
  vector(real(8)) :: v1(10)
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<10x!fir.vector<2:f64>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_r8_arr

! CHECK-LABEL: @vec_lxvp_test_vp_arr_
subroutine vec_lxvp_test_vp_arr(v1, offset, vp)
  implicit none
  integer(8) :: offset
  __vector_pair :: v1(10)
  __vector_pair :: vp
  vp = vec_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i64>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<10x!fir.vector<256:i1>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i64, ptr %1, align 8
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i64 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_vp_arr

!----------------------
! vsx_lxvp
!----------------------
! CHECK-LABEL: @vsx_lxvp_test_i4_
subroutine vsx_lxvp_test_i4(v1, offset, vp)
  implicit none
  integer(2) :: offset
  vector(integer(4)) :: v1
  __vector_pair :: vp
  vp = vsx_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vsx_lxvp_test_i4

! CHECK-LABEL: @vsx_lxvp_test_r8_
subroutine vsx_lxvp_test_r8(v1, offset, vp)
  implicit none
  integer(2) :: offset
  vector(real(8)) :: v1
  __vector_pair :: vp
  vp = vsx_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i16>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.vector<2:f64>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vsx_lxvp_test_r8

! CHECK-LABEL: @vsx_lxvp_test_i2_arr_
subroutine vsx_lxvp_test_i2_arr(v1, offset, vp)
  implicit none
  integer :: offset
  vector(integer(2)) :: v1(10)
  __vector_pair :: vp
  vp = vsx_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i32>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<10x!fir.vector<8:i16>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vsx_lxvp_test_i2_arr

! CHECK-LABEL: @vsx_lxvp_test_vp_arr_
subroutine vsx_lxvp_test_vp_arr(v1, offset, vp)
  implicit none
  integer(8) :: offset
  __vector_pair :: v1(10)
  __vector_pair :: vp
  vp = vsx_lxvp(offset, v1)

! FIR: %[[offset:.*]] = fir.load %arg1 : !fir.ref<i64>
! FIR: %[[v1:.*]] = fir.convert %arg0 : (!fir.ref<!fir.array<10x!fir.vector<256:i1>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[v1]], %[[offset]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.vsx.lxvp(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> !fir.vector<256:i1>
! FIR: fir.store %[[call]] to %arg2 : !fir.ref<!fir.vector<256:i1>>

! LLVMIR-P10: %[[offset:.*]] = load i64, ptr %1, align 8
! LLVMIR-P10: %[[addr:.*]] = getelementptr i8, ptr %0, i64 %[[offset]]
! LLVMIR-P10: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR-P10: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vsx_lxvp_test_vp_arr
