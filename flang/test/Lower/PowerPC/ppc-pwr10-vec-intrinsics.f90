! RUN: %flang_fc1 -triple powerpc64le-unknown-unknown -target-cpu pwr10 -emit-llvm %s -o - | FileCheck --check-prefixes="CHECK" %s
! REQUIRES: target=powerpc{{.*}}
      subroutine test_cvspbf16()
      implicit none
      vector(unsigned(1)) :: v1, v2
      v1 = vec_cvspbf16(v2)
      end subroutine test_cvspbf16

!CHECK-LABEL: @test_cvspbf16_
!CHECK:  %1 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = load <16 x i8>, ptr %2, align 16
!CHECK:  %4 = call <16 x i8> @llvm.ppc.vsx.xvcvspbf16(<16 x i8> %3)
!CHECK:  store <16 x i8> %4, ptr %1, align 16

      subroutine test_cvbf16spn()
      implicit none
      vector(unsigned(1)) :: v1, v2
      v1 = vec_cvbf16spn(v2)
      end subroutine test_cvbf16spn

!CHECK-LABEL: @test_cvbf16spn_
!CHECK:  %1 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %2 = alloca <16 x i8>, i64 1, align 16
!CHECK:  %3 = load <16 x i8>, ptr %2, align 16
!CHECK:  %4 = call <16 x i8> @llvm.ppc.vsx.xvcvbf16spn(<16 x i8> %3)
!CHECK:  store <16 x i8> %4, ptr %1, align 16

!----------------------
! vec_lxvp
!----------------------

      subroutine vec_lxvp_test_i2(v1, offset, vp)
      integer(2) :: offset
      vector(integer(2)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_i2

!CHECK-LABEL: @vec_lxvp_test_i2_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_i4(v1, offset, vp)
      integer(2) :: offset
      vector(integer(4)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_i4

!CHECK-LABEL: @vec_lxvp_test_i4_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_u2(v1, offset, vp)
      integer(2) :: offset
      vector(unsigned(2)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_u2

!CHECK-LABEL: @vec_lxvp_test_u2_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_u4(v1, offset, vp)
      integer(2) :: offset
      vector(unsigned(4)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_u4

!CHECK-LABEL: @vec_lxvp_test_u4_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_r4(v1, offset, vp)
      integer(2) :: offset
      vector(real(4)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_r4

!CHECK-LABEL: @vec_lxvp_test_r4_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_r8(v1, offset, vp)
      integer(2) :: offset
      vector(real(8)) :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_r8

!CHECK-LABEL: @vec_lxvp_test_r8_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_vp(v1, offset, vp)
      integer(2) :: offset
      __vector_pair :: v1
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_vp

!CHECK-LABEL: @vec_lxvp_test_vp_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_i2_arr(v1, offset, vp)
      integer :: offset
      vector(integer(2)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_i2_arr

!CHECK-LABEL: @vec_lxvp_test_i2_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_i4_arr(v1, offset, vp)
      integer :: offset
      vector(integer(4)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_i4_arr

!CHECK-LABEL: @vec_lxvp_test_i4_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_u2_arr(v1, offset, vp)
      integer :: offset
      vector(unsigned(2)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_u2_arr

!CHECK-LABEL: @vec_lxvp_test_u2_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_u4_arr(v1, offset, vp)
      integer :: offset
      vector(unsigned(4)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_u4_arr

!CHECK-LABEL: @vec_lxvp_test_u4_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_r4_arr(v1, offset, vp)
      integer :: offset
      vector(real(4)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_r4_arr

!CHECK-LABEL: @vec_lxvp_test_r4_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_r8_arr(v1, offset, vp)
      integer :: offset
      vector(real(8)) :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_r8_arr

!CHECK-LABEL: @vec_lxvp_test_r8_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vec_lxvp_test_vp_arr(v1, offset, vp)
      integer(8) :: offset
      __vector_pair :: v1(10)
      __vector_pair :: vp
      vp = vec_lxvp(offset, v1)
      end subroutine vec_lxvp_test_vp_arr

!CHECK-LABEL: @vec_lxvp_test_vp_arr_
!CHECK:  %[[offset:.*]] = load i64, ptr %1, align 8
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i64 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

!----------------------
! vsx_lxvp
!----------------------

      subroutine vsx_lxvp_test_i4(v1, offset, vp)
      integer(2) :: offset
      vector(integer(4)) :: v1
      __vector_pair :: vp
      vp = vsx_lxvp(offset, v1)
      end subroutine vsx_lxvp_test_i4

!CHECK-LABEL: @vsx_lxvp_test_i4_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vsx_lxvp_test_r8(v1, offset, vp)
      integer(2) :: offset
      vector(real(8)) :: v1
      __vector_pair :: vp
      vp = vsx_lxvp(offset, v1)
      end subroutine vsx_lxvp_test_r8

!CHECK-LABEL: @vsx_lxvp_test_r8_
!CHECK:  %[[offset:.*]] = load i16, ptr %1, align 2
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vsx_lxvp_test_i2_arr(v1, offset, vp)
      integer :: offset
      vector(integer(2)) :: v1(10)
      __vector_pair :: vp
      vp = vsx_lxvp(offset, v1)
      end subroutine vsx_lxvp_test_i2_arr

!CHECK-LABEL: @vsx_lxvp_test_i2_arr_
!CHECK:  %[[offset:.*]] = load i32, ptr %1, align 4
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32

      subroutine vsx_lxvp_test_vp_arr(v1, offset, vp)
      integer(8) :: offset
      __vector_pair :: v1(10)
      __vector_pair :: vp
      vp = vsx_lxvp(offset, v1)
      end subroutine vsx_lxvp_test_vp_arr

!CHECK-LABEL: @vsx_lxvp_test_vp_arr_
!CHECK:  %[[offset:.*]] = load i64, ptr %1, align 8
!CHECK:  %[[addr:.*]] = getelementptr i8, ptr %0, i64 %[[offset]]
!CHECK:  %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
!CHECK:  store <256 x i1> %[[call]], ptr %2, align 32
