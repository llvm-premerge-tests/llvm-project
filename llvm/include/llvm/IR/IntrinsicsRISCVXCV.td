//===- IntrinsicsRISCVXCV.td - CORE-V intrinsics -----------*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines all of the CORE-V vendor intrinsics for RISC-V.
//
//===----------------------------------------------------------------------===//

class ScalarCoreVBitManipGprGprIntrinsic
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrWillReturn, IntrSpeculatable]>;

class ScalarCoreVBitManipGprIntrinsic
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty],
                [IntrNoMem, IntrWillReturn, IntrSpeculatable]>;

let TargetPrefix = "riscv" in {
    def int_riscv_cv_bitmanip_extract : ScalarCoreVBitManipGprGprIntrinsic;
    def int_riscv_cv_bitmanip_extractu : ScalarCoreVBitManipGprGprIntrinsic;
    def int_riscv_cv_bitmanip_bclr : ScalarCoreVBitManipGprGprIntrinsic;
    def int_riscv_cv_bitmanip_bset : ScalarCoreVBitManipGprGprIntrinsic;

    def int_riscv_cv_bitmanip_insert
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrWillReturn, IntrSpeculatable]>;

    def int_riscv_cv_bitmanip_ff1 : ScalarCoreVBitManipGprIntrinsic;
    def int_riscv_cv_bitmanip_fl1 : ScalarCoreVBitManipGprIntrinsic;
    def int_riscv_cv_bitmanip_clb : ScalarCoreVBitManipGprIntrinsic;
    def int_riscv_cv_bitmanip_cnt : ScalarCoreVBitManipGprIntrinsic;

    def int_riscv_cv_bitmanip_ror : ScalarCoreVBitManipGprGprIntrinsic;

    def int_riscv_cv_bitmanip_bitrev
    : Intrinsic<[llvm_i32_ty], [llvm_i32_ty, llvm_i32_ty, llvm_i32_ty],
                [IntrNoMem, IntrWillReturn, IntrSpeculatable,
                 ImmArg<ArgIndex<1>>, ImmArg<ArgIndex<2>>]>;
} // TargetPrefix = "riscv"
