! RUN: bbc -emit-fir %s -o - | FileCheck --check-prefixes="FIR" %s
! RUN: %flang -emit-llvm -S %s -o - | FileCheck --check-prefixes="LLVMIR" %s
! R: %flang -mcpu=pwr9 -emit-llvm -S %s -o - | FileCheck --check-prefixes="CHECK_P9" %s
! REQUIRES: target=powerpc{{.*}}

!----------------------
! vec_ld
!----------------------

! CHECK-LABEL: @vec_ld_testi8
subroutine vec_ld_testi8(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<16:i8>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<16xi8>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %{{.*}}, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testi8

! CHECK-LABEL: @vec_ld_testi16
subroutine vec_ld_testi16(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<8xi16>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testi16

! CHECK-LABEL: @vec_ld_testi32
subroutine vec_ld_testi32(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[cnv:.*]] = fir.convert %[[call]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[bc:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testi32

! CHECK-LABEL: @vec_ld_testf32
subroutine vec_ld_testf32(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! FIR: %[[arg1i32:.*]] = fir.convert %[[arg1]] : (i64) -> i32
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<4:f32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1i32]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<4xf32>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[arg1i32:.*]] = trunc i64 %[[arg1]] to i32
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1i32]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testf32

! CHECK-LABEL: @vec_ld_testu32
subroutine vec_ld_testu32(arg1, arg2, res)
  integer(1) :: arg1
  vector(unsigned(4)) :: arg2, res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<4:ui32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[cnv:.*]] = fir.convert %[[call]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:ui32>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_ld_testu32

! CHECK-LABEL: @vec_ld_testi32a
subroutine vec_ld_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(10)
  vector(integer(4)) :: res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[cnv:.*]] = fir.convert %[[call]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_ld_testi32a

! CHECK-LABEL: @vec_ld_testf32av
subroutine vec_ld_testf32av(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2(2, 4, 8)
  vector(real(4)) :: res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! FIR: %[[arg1i32:.*]] = fir.convert %[[arg1]] : (i64) -> i32
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<2x4x8x!fir.vector<4:f32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1i32]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<4xf32>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[arg1i32:.*]] = trunc i64 %[[arg1]] to i32
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1i32]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testf32av

! CHECK-LABEL: @vec_ld_testi32s
subroutine vec_ld_testi32s(arg1, arg2, res)
  integer(4) :: arg1
  real(4) :: arg2
  vector(real(4)) :: res
  res = vec_ld(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<4xf32>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ld_testi32s

!----------------------
! vec_lde
!----------------------

! CHECK-LABEL: @vec_lde_testi8s
subroutine vec_lde_testi8s(arg1, arg2, res)
  integer(1) :: arg1
  integer(1) :: arg2
  vector(integer(1)) :: res
  res = vec_lde(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<i8>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvebx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<16xi8>
! FIR: %[[cnv:.*]] = fir.convert %[[call]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <16 x i8> @llvm.ppc.altivec.lvebx(ptr %[[addr]])
! LLVMIR: store <16 x i8> %[[call]], ptr %2, align 16
end subroutine vec_lde_testi8s

! CHECK-LABEL: @vec_lde_testi16a
subroutine vec_lde_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  integer(2) :: arg2(2, 4, 8)
  vector(integer(2)) :: res
  res = vec_lde(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8xi16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvehx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<8xi16>
! FIR: %[[cnv:.*]] = fir.convert %[[call]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <8 x i16> @llvm.ppc.altivec.lvehx(ptr %[[addr]])
! LLVMIR: store <8 x i16> %[[call]], ptr %2, align 16
end subroutine vec_lde_testi16a

! CHECK-LABEL: @vec_lde_testi32a
subroutine vec_lde_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(4)
  vector(integer(4)) :: res
  res = vec_lde(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvewx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[cnv:.*]] = fir.convert %[[call]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvewx(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_lde_testi32a

! CHECK-LABEL: @vec_lde_testf32a
subroutine vec_lde_testf32a(arg1, arg2, res)
  integer(8) :: arg1
  real(4) :: arg2(4)
  vector(real(4)) :: res
  res = vec_lde(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4xf32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvewx(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<4xf32>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvewx(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_lde_testf32a

!----------------------
! vec_ldl
!----------------------

! CHECK-LABEL: @vec_ldl_testi8
subroutine vec_ldl_testi8(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2, res
  res = vec_ldl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<16:i8>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvxl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<16xi8>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %{{.*}}, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testi8

! CHECK-LABEL: @vec_ldl_testi16
subroutine vec_ldl_testi16(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2, res
  res = vec_ldl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvxl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<8xi16>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testi16

! CHECK-LABEL: @vec_ldl_testi32
subroutine vec_ldl_testi32(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2, res
  res = vec_ldl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<4:i32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvxl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[cnv:.*]] = fir.convert %[[call]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[bc:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testi32

! CHECK-LABEL: @vec_ldl_testf32
subroutine vec_ldl_testf32(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2, res
  res = vec_ldl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<4:f32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvxl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<4xf32>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testf32

! CHECK-LABEL: @vec_ldl_testu32
subroutine vec_ldl_testu32(arg1, arg2, res)
  integer(1) :: arg1
  vector(unsigned(4)) :: arg2, res
  res = vec_ldl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.vector<4:ui32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvxl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[cnv:.*]] = fir.convert %[[call]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:ui32>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_ldl_testu32

! CHECK-LABEL: @vec_ldl_testi32a
subroutine vec_ldl_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  integer(4) :: arg2(10)
  vector(integer(4)) :: res
  res = vec_ldl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<10xi32>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvxl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[cnv:.*]] = fir.convert %[[call]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[call]], ptr %2, align 16
end subroutine vec_ldl_testi32a

! CHECK-LABEL: @vec_ldl_testf32av
subroutine vec_ldl_testf32av(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(4)) :: arg2(2, 4, 8)
  vector(real(4)) :: res
  res = vec_ldl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i64>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<!fir.array<2x4x8x!fir.vector<4:f32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvxl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<4xf32>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testf32av

! CHECK-LABEL: @vec_ldl_testi32s
subroutine vec_ldl_testi32s(arg1, arg2, res)
  integer(4) :: arg1
  real(4) :: arg2
  vector(real(4)) :: res
  res = vec_ldl(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %{{.*}} : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %{{.*}} : (!fir.ref<f32>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[call:.*]] = fir.call @llvm.ppc.altivec.lvxl(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[call]] : vector<4xi32> to vector<4xf32>
! FIR: %[[cnv:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[cnv]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[call:.*]] = call <4 x i32> @llvm.ppc.altivec.lvxl(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <4 x i32> %[[call]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_ldl_testi32s

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

! LLVMIR: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i16, ptr %1, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i16 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i32, ptr %1, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i32 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
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

! LLVMIR: %[[offset:.*]] = load i64, ptr %1, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %0, i64 %[[offset]]
! LLVMIR: %[[call:.*]] = call <256 x i1> @llvm.ppc.vsx.lxvp(ptr %[[addr]])
! LLVMIR: store <256 x i1> %[[call]], ptr %2, align 32
end subroutine vec_lxvp_test_vp_arr

!----------------------
! vec_xld2
!----------------------

! CHECK-LABEL: @vec_xld2_testi8a
subroutine vec_xld2_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2(4)
  vector(integer(1)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<16:i8>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi8a

! CHECK-LABEL: @vec_xld2_testi16
subroutine vec_xld2_testi16(arg1, arg2, res)
  integer :: arg1
  vector(integer(2)) :: arg2
  vector(integer(2)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.vector<8:i16>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<8xi16>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi16

! CHECK-LABEL: @vec_xld2_testi32a
subroutine vec_xld2_testi32a(arg1, arg2, res)
  integer(4) :: arg1
  vector(integer(4)) :: arg2(41)
  vector(integer(4)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<41x!fir.vector<4:i32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<4xi32>) -> !fir.vector<4:i32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:i32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <4 x i32>
! LLVMIR: store <4 x i32> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi32a

! CHECK-LABEL: @vec_xld2_testi64a
subroutine vec_xld2_testi64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(integer(8)) :: arg2(4)
  vector(integer(8)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<2:i64>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<2xi64>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<2xi64>) -> !fir.vector<2:i64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:i64>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <2 x i64>
! LLVMIR: store <2 x i64> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testi64a

! CHECK-LABEL: @vec_xld2_testf32a
subroutine vec_xld2_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  vector(real(4)) :: arg2(4)
  vector(real(4)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<4:f32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<2xf64> to vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: %[[bc:.*]] = bitcast <2 x double> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[bc]], ptr %2, align 16
end subroutine vec_xld2_testf32a

! CHECK-LABEL: @vec_xld2_testf64a
subroutine vec_xld2_testf64a(arg1, arg2, res)
  integer(8) :: arg1
  vector(real(8)) :: arg2(4)
  vector(real(8)) :: res
  res = vec_xld2(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i64>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<2:f64>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i64) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvd2x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<2xf64>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<2xf64>) -> !fir.vector<2:f64>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<2:f64>>

! LLVMIR: %[[arg1:.*]] = load i64, ptr %0, align 8
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i64 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call contract <2 x double> @llvm.ppc.vsx.lxvd2x(ptr %[[addr]])
! LLVMIR: store <2 x double> %[[ld]], ptr %2, align 16
end subroutine vec_xld2_testf64a

!----------------------
! vec_xlw4
!----------------------

! CHECK-LABEL: @vec_xlw4_testi8a
subroutine vec_xlw4_testi8a(arg1, arg2, res)
  integer(1) :: arg1
  vector(integer(1)) :: arg2(2, 4, 8)
  vector(integer(1)) :: res
  res = vec_xlw4(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i8>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8x!fir.vector<16:i8>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i8) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<16xi8>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<16xi8>) -> !fir.vector<16:i8>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<16:i8>>

! LLVMIR: %[[arg1:.*]] = load i8, ptr %0, align 1
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i8 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: %[[res:.*]] = bitcast <4 x i32> %[[ld]] to <16 x i8>
! LLVMIR: store <16 x i8> %[[res]], ptr %2, align 16
end subroutine vec_xlw4_testi8a

! CHECK-LABEL: @vec_xlw4_testi16a
subroutine vec_xlw4_testi16a(arg1, arg2, res)
  integer(2) :: arg1
  vector(integer(2)) :: arg2(2, 4, 8)
  vector(integer(2)) :: res
  res = vec_xlw4(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8x!fir.vector<8:i16>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<8xi16>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<8xi16>) -> !fir.vector<8:i16>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<8:i16>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: %[[res:.*]] = bitcast <4 x i32> %[[ld]] to <8 x i16>
! LLVMIR: store <8 x i16> %[[res]], ptr %2, align 16
end subroutine vec_xlw4_testi16a

! CHECK-LABEL: @vec_xlw4_testu32a
subroutine vec_xlw4_testu32a(arg1, arg2, res)
  integer(4) :: arg1
  vector(unsigned(4)) :: arg2(2, 4, 8)
  vector(unsigned(4)) :: res
  res = vec_xlw4(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i32>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<2x4x8x!fir.vector<4:ui32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i32) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[res:.*]] = fir.convert %[[ld]] : (vector<4xi32>) -> !fir.vector<4:ui32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:ui32>>

! LLVMIR: %[[arg1:.*]] = load i32, ptr %0, align 4
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i32 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: store <4 x i32> %[[ld]], ptr %2, align 16
end subroutine vec_xlw4_testu32a

! CHECK-LABEL: @vec_xlw4_testf32a
subroutine vec_xlw4_testf32a(arg1, arg2, res)
  integer(2) :: arg1
  vector(real(4)) :: arg2(4)
  vector(real(4)) :: res
  res = vec_xlw4(arg1, arg2)

! FIR: %[[arg1:.*]] = fir.load %arg0 : !fir.ref<i16>
! FIR: %[[arg2:.*]] = fir.convert %arg1 : (!fir.ref<!fir.array<4x!fir.vector<4:f32>>>) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[addr:.*]] = fir.coordinate_of %[[arg2]], %[[arg1]] : (!fir.ref<!fir.array<?xi8>>, i16) -> !fir.ref<!fir.array<?xi8>>
! FIR: %[[ld:.*]] = fir.call @llvm.ppc.vsx.lxvw4x(%[[addr]]) fastmath<contract> : (!fir.ref<!fir.array<?xi8>>) -> vector<4xi32>
! FIR: %[[bc:.*]] = vector.bitcast %[[ld]] : vector<4xi32> to vector<4xf32>
! FIR: %[[res:.*]] = fir.convert %[[bc]] : (vector<4xf32>) -> !fir.vector<4:f32>
! FIR: fir.store %[[res]] to %arg2 : !fir.ref<!fir.vector<4:f32>>

! LLVMIR: %[[arg1:.*]] = load i16, ptr %0, align 2
! LLVMIR: %[[addr:.*]] = getelementptr i8, ptr %1, i16 %[[arg1]]
! LLVMIR: %[[ld:.*]] = call <4 x i32> @llvm.ppc.vsx.lxvw4x(ptr %[[addr]])
! LLVMIR: %[[res:.*]] = bitcast <4 x i32> %[[ld]] to <4 x float>
! LLVMIR: store <4 x float> %[[res]], ptr %2, align 16
end subroutine vec_xlw4_testf32a
