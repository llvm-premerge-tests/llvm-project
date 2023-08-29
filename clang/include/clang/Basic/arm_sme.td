//===--- arm_sme.td - ARM SME compiler interface ------------------------===//
//
//  Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
//  See https://llvm.org/LICENSE.txt for license information.
//  SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
//  This file defines the TableGen definitions from which the ARM SME header
//  file will be generated.  See:
//
//      https://developer.arm.com/architectures/system-architectures/software-standards/acle
//
//===----------------------------------------------------------------------===//

include "arm_sve_sme_incl.td"

class Inst<string n, string p, string t, string i,
           list<FlagType> ft, list<ImmCheck> ch, MemEltType met = MemEltTyDefault> {
  string Name = n;
  string Prototype = p;
  string Types = t;
  string TargetGuard = "sme";
  int Merge = MergeNone.Value;
  string MergeSuffix = MergeNone.Suffix;
  string LLVMIntrinsic = i;
  list<FlagType> Flags = ft;
  list<ImmCheck> ImmChecks = ch;
  int MemEltType = met.Value;
}

// SInst: Instruction with signed/unsigned suffix (e.g., "s8", "u8")
class SInst<string n, string p, string t, string i = "",
            list<FlagType> ft = [], list<ImmCheck> ch = []>
    : Inst<n, p, t, i, ft, ch, MemEltTyDefault> {
}

// MInst: Instructions which access memory
class MInst<string n, string p, string t, list<FlagType> f,
            MemEltType met = MemEltTyDefault, string i = "",
            list<ImmCheck> ch = []>
    : Inst<n, p, t, i, f, ch, met> {
}

////////////////////////////////////////////////////////////////////////////////
// Loads

multiclass ZALoad<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : MInst<"svld1_hor_" # n_suffix, "vimiPQ", t,
                          [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                          MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _H_VNUM : MInst<"svld1_hor_vnum_" # n_suffix, "vimiPQl", t,
                               [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                               MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _V : MInst<"svld1_ver_" # n_suffix, "vimiPQ", t,
                          [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                          MemEltTyDefault, i_prefix # "_vert", ch>;

    def NAME # _V_VNUM : MInst<"svld1_ver_vnum_" # n_suffix, "vimiPQl", t,
                               [IsLoad, IsOverloadNone, IsStreaming, IsSharedZA],
                               MemEltTyDefault, i_prefix # "_vert", ch>;
  }
}

defm SVLD1_ZA8 : ZALoad<"za8", "c", "aarch64_sme_ld1b", [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_15>]>;
defm SVLD1_ZA16 : ZALoad<"za16", "s", "aarch64_sme_ld1h", [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_7>]>;
defm SVLD1_ZA32 : ZALoad<"za32", "i", "aarch64_sme_ld1w", [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_3>]>;
defm SVLD1_ZA64 : ZALoad<"za64", "l", "aarch64_sme_ld1d", [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_1>]>;
defm SVLD1_ZA128 : ZALoad<"za128", "q", "aarch64_sme_ld1q", [ImmCheck<0, ImmCheck0_15>, ImmCheck<2, ImmCheck0_0>]>;

def SVLDR_VNUM_ZA : MInst<"svldr_vnum_za", "vmiQ", "",
                          [IsOverloadNone, IsStreamingCompatible, IsSharedZA],
                          MemEltTyDefault, "aarch64_sme_ldr",
                          [ImmCheck<1, ImmCheck0_15>]>;

////////////////////////////////////////////////////////////////////////////////
// Stores

multiclass ZAStore<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : MInst<"svst1_hor_" # n_suffix, "vimiP%", t,
                          [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                          MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _H_VNUM : MInst<"svst1_hor_vnum_" # n_suffix, "vimiP%l", t,
                               [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                               MemEltTyDefault, i_prefix # "_horiz", ch>;

    def NAME # _V : MInst<"svst1_ver_" # n_suffix, "vimiP%", t,
                          [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                          MemEltTyDefault, i_prefix # "_vert", ch>;

    def NAME # _V_VNUM : MInst<"svst1_ver_vnum_" # n_suffix, "vimiP%l", t,
                               [IsStore, IsOverloadNone, IsStreaming, IsSharedZA, IsPreservesZA],
                               MemEltTyDefault, i_prefix # "_vert", ch>;
  }
}

defm SVST1_ZA8 : ZAStore<"za8", "c", "aarch64_sme_st1b", [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_15>]>;
defm SVST1_ZA16 : ZAStore<"za16", "s", "aarch64_sme_st1h", [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_7>]>;
defm SVST1_ZA32 : ZAStore<"za32", "i", "aarch64_sme_st1w", [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_3>]>;
defm SVST1_ZA64 : ZAStore<"za64", "l", "aarch64_sme_st1d", [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_1>]>;
defm SVST1_ZA128 : ZAStore<"za128", "q", "aarch64_sme_st1q", [ImmCheck<0, ImmCheck0_15>, ImmCheck<2, ImmCheck0_0>]>;

def SVSTR_VNUM_ZA : MInst<"svstr_vnum_za", "vmi%", "",
                          [IsOverloadNone, IsStreamingCompatible, IsSharedZA, IsPreservesZA],
                          MemEltTyDefault, "aarch64_sme_str",
                          [ImmCheck<1, ImmCheck0_15>]>;

////////////////////////////////////////////////////////////////////////////////
// Read horizontal/vertical ZA slices

multiclass ZARead<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : SInst<"svread_hor_" # n_suffix # "[_{d}]_m", "ddPimi", t,
                          i_prefix # "_horiz",
                          [IsReadZA, IsStreaming, IsSharedZA, IsPreservesZA], ch>;

    def NAME # _V : SInst<"svread_ver_" # n_suffix # "[_{d}]_m", "ddPimi", t,
                          i_prefix # "_vert",
                          [IsReadZA, IsStreaming, IsSharedZA, IsPreservesZA], ch>;
  }
}

defm SVREAD_ZA8 : ZARead<"za8", "cUc", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_0>, ImmCheck<4, ImmCheck0_15>]>;
defm SVREAD_ZA16 : ZARead<"za16", "sUshb", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_1>, ImmCheck<4, ImmCheck0_7>]>;
defm SVREAD_ZA32 : ZARead<"za32", "iUif", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_3>, ImmCheck<4, ImmCheck0_3>]>;
defm SVREAD_ZA64 : ZARead<"za64", "lUld", "aarch64_sme_read", [ImmCheck<2, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
defm SVREAD_ZA128 : ZARead<"za128", "csilUcUsUiUlhbfd", "aarch64_sme_readq", [ImmCheck<2, ImmCheck0_15>, ImmCheck<4, ImmCheck0_0>]>;

////////////////////////////////////////////////////////////////////////////////
// Write horizontal/vertical ZA slices

multiclass ZAWrite<string n_suffix, string t, string i_prefix, list<ImmCheck> ch> {
  let TargetGuard = "sme" in {
    def NAME # _H : SInst<"svwrite_hor_" # n_suffix # "[_{d}]_m", "vimiPd", t,
                          i_prefix # "_horiz",
                          [IsWriteZA, IsStreaming, IsSharedZA], ch>;

    def NAME # _V : SInst<"svwrite_ver_" # n_suffix # "[_{d}]_m", "vimiPd", t,
                          i_prefix # "_vert",
                          [IsWriteZA, IsStreaming, IsSharedZA], ch>;
  }
}

defm SVWRITE_ZA8 : ZAWrite<"za8", "cUc", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_15>]>;
defm SVWRITE_ZA16 : ZAWrite<"za16", "sUshb", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_7>]>;
defm SVWRITE_ZA32 : ZAWrite<"za32", "iUif", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_3>]>;
defm SVWRITE_ZA64 : ZAWrite<"za64", "lUld", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_1>]>;
defm SVWRITE_ZA128 : ZAWrite<"za128", "csilUcUsUiUlhbfd", "aarch64_sme_writeq", [ImmCheck<0, ImmCheck0_15>, ImmCheck<2, ImmCheck0_0>]>;

////////////////////////////////////////////////////////////////////////////////
// SME - Zero

let TargetGuard = "sme" in {
  def SVZERO_MASK_ZA : SInst<"svzero_mask_za", "vi", "", "aarch64_sme_zero",
                             [IsOverloadNone, IsStreamingCompatible, IsSharedZA],
                             [ImmCheck<0, ImmCheck0_255>]>;
  def SVZERO_ZA      : SInst<"svzero_za", "v", "", "aarch64_sme_zero",
                             [IsOverloadNone, IsStreamingCompatible, IsSharedZA]>;
}

////////////////////////////////////////////////////////////////////////////////
// SME - Counting elements in a streaming vector

multiclass ZACount<string n_suffix> {
  let TargetGuard = "sme" in {
    def NAME : SInst<"sv" # n_suffix, "nv", "",
                      "aarch64_sme_" # n_suffix,
                      [IsOverloadNone, IsStreamingCompatible, IsPreservesZA]>;
  }
}

defm SVCNTSB : ZACount<"cntsb">;
defm SVCNTSH : ZACount<"cntsh">;
defm SVCNTSW : ZACount<"cntsw">;
defm SVCNTSD : ZACount<"cntsd">;

////////////////////////////////////////////////////////////////////////////////
// SME - ADDHA/ADDVA

multiclass ZAAdd<string n_suffix> {
  let TargetGuard = "sme" in {
    def NAME # _ZA32: SInst<"sv" # n_suffix # "_za32[_{d}]_m", "viPPd", "iUi",
                      "aarch64_sme_" # n_suffix, [IsStreaming, IsSharedZA],
                      [ImmCheck<0, ImmCheck0_3>]>;
  }

  let TargetGuard = "sme-i16i64" in {
    def NAME # _ZA64: SInst<"sv" # n_suffix # "_za64[_{d}]_m", "viPPd", "lUl",
                     "aarch64_sme_" # n_suffix, [IsStreaming, IsSharedZA],
                     [ImmCheck<0, ImmCheck0_7>]>;
  }
}

defm SVADDHA : ZAAdd<"addha">;
defm SVADDVA : ZAAdd<"addva">;

////////////////////////////////////////////////////////////////////////////////
// SME - SMOPA, SMOPS, UMOPA, UMOPS

multiclass ZAIntOuterProd<string n_suffix1, string n_suffix2> {
  let TargetGuard = "sme" in {
    def NAME # _ZA32_B: SInst<"sv" # n_suffix2 # "_za32[_{d}]_m",
                              "viPPdd", !cond(!eq(n_suffix1, "s") : "", true: "U") # "c",
                              "aarch64_sme_" # n_suffix1 # n_suffix2 # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;
  }

  let TargetGuard = "sme-i16i64" in {
    def NAME # _ZA64_H: SInst<"sv" # n_suffix2 # "_za64[_{d}]_m",
                              "viPPdd", !cond(!eq(n_suffix1, "s") : "", true: "U") # "s",
                              "aarch64_sme_" # n_suffix1 # n_suffix2 # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_7>]>;
  }
}

defm SVSMOPA : ZAIntOuterProd<"s", "mopa">;
defm SVSMOPS : ZAIntOuterProd<"s", "mops">;
defm SVUMOPA : ZAIntOuterProd<"u", "mopa">;
defm SVUMOPS : ZAIntOuterProd<"u", "mops">;

////////////////////////////////////////////////////////////////////////////////
// SME - SUMOPA, SUMOPS, USMOPA, USMOPS

multiclass ZAIntOuterProdMixedSigns<string n_suffix1, string n_suffix2> {
  let TargetGuard = "sme" in {
    def NAME # _ZA32_B: SInst<"sv" # n_suffix1 # n_suffix2 # "_za32[_{d}]_m",
                              "viPPd" # !cond(!eq(n_suffix1, "su") : "u", true: "x"),
                              !cond(!eq(n_suffix1, "su") : "", true: "U") # "c",
                              "aarch64_sme_" # n_suffix1 # n_suffix2 # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;
  }

  let TargetGuard = "sme-i16i64" in {
    def NAME # _ZA64_H: SInst<"sv" # n_suffix1 # n_suffix2 # "_za64[_{d}]_m",
                              "viPPd" # !cond(!eq(n_suffix1, "su") : "u", true: "x"),
                              !cond(!eq(n_suffix1, "su") : "", true: "U") # "s",
                              "aarch64_sme_" # n_suffix1 # n_suffix2 # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_7>]>;
  }
}

defm SVSUMOPA : ZAIntOuterProdMixedSigns<"su", "mopa">;
defm SVSUMOPS : ZAIntOuterProdMixedSigns<"su", "mops">;
defm SVUSMOPA : ZAIntOuterProdMixedSigns<"us", "mopa">;
defm SVUSMOPS : ZAIntOuterProdMixedSigns<"us", "mops">;

////////////////////////////////////////////////////////////////////////////////
// SME - FMOPA, FMOPS

multiclass ZAFPOuterProd<string n_suffix> {
  let TargetGuard = "sme" in {
    def NAME # _ZA32_B: SInst<"sv" # n_suffix # "_za32[_{d}]_m", "viPPdd", "h",
                              "aarch64_sme_" # n_suffix # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;

    def NAME # _ZA32_H: SInst<"sv" # n_suffix # "_za32[_{d}]_m", "viPPdd", "b",
                              "aarch64_sme_" # n_suffix # "_wide",
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;

    def NAME # _ZA32_S: SInst<"sv" # n_suffix # "_za32[_{d}]_m", "viPPdd", "f",
                              "aarch64_sme_" # n_suffix,
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_3>]>;
  }

  let TargetGuard = "sme-f64f64" in {
    def NAME # _ZA64_D: SInst<"sv" # n_suffix # "_za64[_{d}]_m", "viPPdd", "d",
                              "aarch64_sme_" # n_suffix,
                              [IsStreaming, IsSharedZA],
                              [ImmCheck<0, ImmCheck0_7>]>;
  }
}

defm SVMOPA : ZAFPOuterProd<"mopa">;
defm SVMOPS : ZAFPOuterProd<"mops">;

// FMLA/FMLS
let TargetGuard = "sme2" in {
  def SVMLA_MULTI_VG1x2_F32 : Inst<"svmla_za32[_{d}]_vg1x2", "vmi22", "f", "aarch64_sme_fmla_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLA_MULTI_VG1x4_F32 : Inst<"svmla_za32[_{d}]_vg1x4", "vmi44", "f", "aarch64_sme_fmla_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLS_MULTI_VG1x2_F32 : Inst<"svmls_za32[_{d}]_vg1x2", "vmi22", "f", "aarch64_sme_fmls_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLS_MULTI_VG1x4_F32 : Inst<"svmls_za32[_{d}]_vg1x4", "vmi44", "f", "aarch64_sme_fmls_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  def SVMLA_SINGLE_VG1x2_F32 : Inst<"svmla[_single]_za32[_{d}]_vg1x2", "vmi2d", "f", "aarch64_sme_fmla_single_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLA_SINGLE_VG1x4_F32 : Inst<"svmla[_single]_za32[_{d}]_vg1x4", "vmi4d", "f", "aarch64_sme_fmla_single_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLS_SINGLE_VG1x2_F32 : Inst<"svmls[_single]_za32[_{d}]_vg1x2", "vmi2d", "f", "aarch64_sme_fmls_single_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLS_SINGLE_VG1x4_F32 : Inst<"svmls[_single]_za32[_{d}]_vg1x4", "vmi4d", "f", "aarch64_sme_fmls_single_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  def SVMLA_LANE_VG1x2_F32 : Inst<"svmla_lane_za32[_{d}]_vg1x2", "vmi2di", "f", "aarch64_sme_fmla_lane_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVMLA_LANE_VG1x4_F32 : Inst<"svmla_lane_za32[_{d}]_vg1x4", "vmi4di", "f", "aarch64_sme_fmla_lane_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVMLS_LANE_VG1x2_F32 : Inst<"svmls_lane_za32[_{d}]_vg1x2", "vmi2di", "f", "aarch64_sme_fmls_lane_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVMLS_LANE_VG1x4_F32 : Inst<"svmls_lane_za32[_{d}]_vg1x4", "vmi4di", "f", "aarch64_sme_fmls_lane_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
}

let TargetGuard = "sme2,sme-f64f64" in {
  def SVMLA_MULTI_VG1x2_F64 : Inst<"svmla_za64[_{d}]_vg1x2", "vmi22", "d", "aarch64_sme_fmla_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLA_MULTI_VG1x4_F64 : Inst<"svmla_za64[_{d}]_vg1x4", "vmi44", "d", "aarch64_sme_fmla_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLS_MULTI_VG1x2_F64 : Inst<"svmls_za64[_{d}]_vg1x2", "vmi22", "d", "aarch64_sme_fmls_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLS_MULTI_VG1x4_F64 : Inst<"svmls_za64[_{d}]_vg1x4", "vmi44", "d", "aarch64_sme_fmls_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  def SVMLA_SINGLE_VG1x2_F64 : Inst<"svmla[_single]_za64[_{d}]_vg1x2", "vmi2d", "d", "aarch64_sme_fmla_single_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLA_SINGLE_VG1x4_F64 : Inst<"svmla[_single]_za64[_{d}]_vg1x4", "vmi4d", "d", "aarch64_sme_fmla_single_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLS_SINGLE_VG1x2_F64 : Inst<"svmls[_single]_za64[_{d}]_vg1x2", "vmi2d", "d", "aarch64_sme_fmls_single_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVMLS_SINGLE_VG1x4_F64 : Inst<"svmls[_single]_za64[_{d}]_vg1x4", "vmi4d", "d", "aarch64_sme_fmls_single_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  def SVMLA_LANE_VG1x2_F64 : Inst<"svmla_lane_za64[_{d}]_vg1x2", "vmi2di", "d", "aarch64_sme_fmla_lane_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
  def SVMLA_LANE_VG1x4_F64 : Inst<"svmla_lane_za64[_{d}]_vg1x4", "vmi4di", "d", "aarch64_sme_fmla_lane_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
  def SVMLS_LANE_VG1x2_F64 : Inst<"svmls_lane_za64[_{d}]_vg1x2", "vmi2di", "d", "aarch64_sme_fmls_lane_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
  def SVMLS_LANE_VG1x4_F64 : Inst<"svmls_lane_za64[_{d}]_vg1x4", "vmi4di", "d", "aarch64_sme_fmls_lane_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
}

// FMLAL/FMLSL/UMLAL/SMLAL
// SMLALL/UMLALL/USMLALL/SUMLALL
let TargetGuard = "sme2" in {
  // MULTI MLAL
  def SVMLAL_MULTI_VG2x2_F16 : Inst<"svmla_za32[_{d}]_vg2x2", "vmi22", "bh", "aarch64_sme_fmlal_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_MULTI_VG2x4_F16 : Inst<"svmla_za32[_{d}]_vg2x4", "vmi44", "bh", "aarch64_sme_fmlal_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_MULTI_VG2x2_S16 : Inst<"svmla_za32[_{d}]_vg2x2", "vmi22", "s",  "aarch64_sme_smlal_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_MULTI_VG2x4_S16 : Inst<"svmla_za32[_{d}]_vg2x4", "vmi44", "s",  "aarch64_sme_smlal_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_MULTI_VG2x2_U16 : Inst<"svmla_za32[_{d}]_vg2x2", "vmi22", "Us", "aarch64_sme_umlal_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_MULTI_VG2x4_U16 : Inst<"svmla_za32[_{d}]_vg2x4", "vmi44", "Us", "aarch64_sme_umlal_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;

  def SVMLAL_MULTI_VG4x2_S8 : Inst<"svmla_za32[_{d}]_vg4x2", "vmi22", "c",  "aarch64_sme_smla_za32_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLAL_MULTI_VG4x2_U8 : Inst<"svmla_za32[_{d}]_vg4x2", "vmi22", "Uc", "aarch64_sme_umla_za32_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLAL_MULTI_VG4x4_S8 : Inst<"svmla_za32[_{d}]_vg4x4", "vmi44", "c",  "aarch64_sme_smla_za32_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLAL_MULTI_VG4x4_U8 : Inst<"svmla_za32[_{d}]_vg4x4", "vmi44", "Uc", "aarch64_sme_umla_za32_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;

  let TargetGuard = "sme-i16i64" in {
    def SVMLAL_MULTI_VG4x2_S16 : Inst<"svmla_za64[_{d}]_vg4x2", "vmi22", "s",  "aarch64_sme_smla_za64_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLAL_MULTI_VG4x2_U16 : Inst<"svmla_za64[_{d}]_vg4x2", "vmi22", "Us", "aarch64_sme_umla_za64_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLAL_MULTI_VG4x4_S16 : Inst<"svmla_za64[_{d}]_vg4x4", "vmi44", "s",  "aarch64_sme_smla_za64_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLAL_MULTI_VG4x4_U16 : Inst<"svmla_za64[_{d}]_vg4x4", "vmi44", "Us", "aarch64_sme_umla_za64_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  }

  // MULTI MLSL
  def SVMLSL_MULTI_VG2x2_F16 : Inst<"svmls_za32[_{d}]_vg2x2", "vmi22", "bh", "aarch64_sme_fmlsl_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_MULTI_VG2x4_F16 : Inst<"svmls_za32[_{d}]_vg2x4", "vmi44", "bh", "aarch64_sme_fmlsl_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_MULTI_VG2x2_S16 : Inst<"svmls_za32[_{d}]_vg2x2", "vmi22", "s",  "aarch64_sme_smlsl_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_MULTI_VG2x4_S16 : Inst<"svmls_za32[_{d}]_vg2x4", "vmi44", "s",  "aarch64_sme_smlsl_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_MULTI_VG2x2_U16 : Inst<"svmls_za32[_{d}]_vg2x2", "vmi22", "Us", "aarch64_sme_umlsl_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_MULTI_VG2x4_U16 : Inst<"svmls_za32[_{d}]_vg2x4", "vmi44", "Us", "aarch64_sme_umlsl_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;

  def SVMLSL_MULTI_VG4x2_S8 : Inst<"svmls_za32[_{d}]_vg4x2", "vmi22", "c",  "aarch64_sme_smls_za32_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLSL_MULTI_VG4x2_U8 : Inst<"svmls_za32[_{d}]_vg4x2", "vmi22", "Uc", "aarch64_sme_umls_za32_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLSL_MULTI_VG4x4_S8 : Inst<"svmls_za32[_{d}]_vg4x4", "vmi44", "c",  "aarch64_sme_smls_za32_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLSL_MULTI_VG4x4_U8 : Inst<"svmls_za32[_{d}]_vg4x4", "vmi44", "Uc", "aarch64_sme_umls_za32_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;

let TargetGuard = "sme-i16i64" in {
    def SVMLSL_MULTI_VG4x2_S16 : Inst<"svmls_za64[_{d}]_vg4x2", "vmi22", "s",  "aarch64_sme_smls_za64_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLSL_MULTI_VG4x2_U16 : Inst<"svmls_za64[_{d}]_vg4x2", "vmi22", "Us", "aarch64_sme_umls_za64_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLSL_MULTI_VG4x4_S16 : Inst<"svmls_za64[_{d}]_vg4x4", "vmi44", "s",  "aarch64_sme_smls_za64_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLSL_MULTI_VG4x4_U16 : Inst<"svmls_za64[_{d}]_vg4x4", "vmi44", "Us", "aarch64_sme_umls_za64_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  }

  // SINGLE MLAL
  def SVMLAL_SINGLE_VG2x1_F16 : Inst<"svmla[_single]_za32[_{d}]_vg2x1", "vmidd", "bh", "aarch64_sme_fmlal_single_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>]>;
  def SVMLAL_SINGLE_VG2x2_F16 : Inst<"svmla[_single]_za32[_{d}]_vg2x2", "vmi2d", "bh", "aarch64_sme_fmlal_single_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_SINGLE_VG2x4_F16 : Inst<"svmla[_single]_za32[_{d}]_vg2x4", "vmi4d", "bh", "aarch64_sme_fmlal_single_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_SINGLE_VG2x1_S16 : Inst<"svmla[_single]_za32[_{d}]_vg2x1", "vmidd", "s",  "aarch64_sme_smlal_single_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>]>;
  def SVMLAL_SINGLE_VG2x2_S16 : Inst<"svmla[_single]_za32[_{d}]_vg2x2", "vmi2d", "s",  "aarch64_sme_smlal_single_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_SINGLE_VG2x4_S16 : Inst<"svmla[_single]_za32[_{d}]_vg2x4", "vmi4d", "s",  "aarch64_sme_smlal_single_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_SINGLE_VG2x1_U16 : Inst<"svmla[_single]_za32[_{d}]_vg2x1", "vmidd", "Us", "aarch64_sme_umlal_single_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>]>;
  def SVMLAL_SINGLE_VG2x2_U16 : Inst<"svmla[_single]_za32[_{d}]_vg2x2", "vmi2d", "Us", "aarch64_sme_umlal_single_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLAL_SINGLE_VG2x4_U16 : Inst<"svmla[_single]_za32[_{d}]_vg2x4", "vmi4d", "Us", "aarch64_sme_umlal_single_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;

  def SVMLAL_SINGLE_VG4x1_S8  : Inst<"svmla[_single]_za32[_{d}]_vg4x1", "vmidd", "c",  "aarch64_sme_smla_za32_single_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>]>;
  def SVMLAL_SINGLE_VG4x1_U8  : Inst<"svmla[_single]_za32[_{d}]_vg4x1", "vmidd", "Uc", "aarch64_sme_umla_za32_single_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>]>;
  def SVMLAL_SINGLE_VG4x2_S8  : Inst<"svmla[_single]_za32[_{d}]_vg4x2", "vmi2d", "c",  "aarch64_sme_smla_za32_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLAL_SINGLE_VG4x2_U8  : Inst<"svmla[_single]_za32[_{d}]_vg4x2", "vmi2d", "Uc", "aarch64_sme_umla_za32_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLAL_SINGLE_VG4x4_S8  : Inst<"svmla[_single]_za32[_{d}]_vg4x4", "vmi4d", "c",  "aarch64_sme_smla_za32_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLAL_SINGLE_VG4x4_U8  : Inst<"svmla[_single]_za32[_{d}]_vg4x4", "vmi4d", "Uc", "aarch64_sme_umla_za32_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;

  let TargetGuard = "sme-i16i64" in {
    def SVMLAL_SINGLE_VG4x1_S16 : Inst<"svmla[_single]_za64[_{d}]_vg4x1", "vmidd", "s",  "aarch64_sme_smla_za64_single_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>]>;
    def SVMLAL_SINGLE_VG4x1_U16 : Inst<"svmla[_single]_za64[_{d}]_vg4x1", "vmidd", "Us", "aarch64_sme_umla_za64_single_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>]>;
    def SVMLAL_SINGLE_VG4x2_S16 : Inst<"svmla[_single]_za64[_{d}]_vg4x2", "vmi2d", "s",  "aarch64_sme_smla_za64_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLAL_SINGLE_VG4x2_U16 : Inst<"svmla[_single]_za64[_{d}]_vg4x2", "vmi2d", "Us", "aarch64_sme_umla_za64_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLAL_SINGLE_VG4x4_S16 : Inst<"svmla[_single]_za64[_{d}]_vg4x4", "vmi4d", "s",  "aarch64_sme_smla_za64_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLAL_SINGLE_VG4x4_U16 : Inst<"svmla[_single]_za64[_{d}]_vg4x4", "vmi4d", "Us", "aarch64_sme_umla_za64_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  }

  // SINGLE MLSL
  def SVMLSL_SINGLE_VG2x1_F16 : Inst<"svmls[_single]_za32[_{d}]_vg2x1", "vmidd", "bh", "aarch64_sme_fmlsl_single_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>]>;
  def SVMLSL_SINGLE_VG2x2_F16 : Inst<"svmls[_single]_za32[_{d}]_vg2x2", "vmi2d", "bh", "aarch64_sme_fmlsl_single_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_SINGLE_VG2x4_F16 : Inst<"svmls[_single]_za32[_{d}]_vg2x4", "vmi4d", "bh", "aarch64_sme_fmlsl_single_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_SINGLE_VG2x1_S16 : Inst<"svmls[_single]_za32[_{d}]_vg2x1", "vmidd", "s",  "aarch64_sme_smlsl_single_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>]>;
  def SVMLSL_SINGLE_VG2x2_S16 : Inst<"svmls[_single]_za32[_{d}]_vg2x2", "vmi2d", "s",  "aarch64_sme_smlsl_single_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_SINGLE_VG2x4_S16 : Inst<"svmls[_single]_za32[_{d}]_vg2x4", "vmi4d", "s",  "aarch64_sme_smlsl_single_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_SINGLE_VG2x1_U16 : Inst<"svmls[_single]_za32[_{d}]_vg2x1", "vmidd", "Us", "aarch64_sme_umlsl_single_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>]>;
  def SVMLSL_SINGLE_VG2x2_U16 : Inst<"svmls[_single]_za32[_{d}]_vg2x2", "vmi2d", "Us", "aarch64_sme_umlsl_single_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;
  def SVMLSL_SINGLE_VG2x4_U16 : Inst<"svmls[_single]_za32[_{d}]_vg2x4", "vmi4d", "Us", "aarch64_sme_umlsl_single_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>]>;

  def SVMLSL_SINGLE_VG4x1_S8  : Inst<"svmls[_single]_za32[_{d}]_vg4x1", "vmidd", "c",  "aarch64_sme_smls_za32_single_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>]>;
  def SVMLSL_SINGLE_VG4x1_U8  : Inst<"svmls[_single]_za32[_{d}]_vg4x1", "vmidd", "Uc", "aarch64_sme_umls_za32_single_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>]>;
  def SVMLSL_SINGLE_VG4x2_S8  : Inst<"svmls[_single]_za32[_{d}]_vg4x2", "vmi2d", "c",  "aarch64_sme_smls_za32_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLSL_SINGLE_VG4x2_U8  : Inst<"svmls[_single]_za32[_{d}]_vg4x2", "vmi2d", "Uc", "aarch64_sme_umls_za32_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLSL_SINGLE_VG4x4_S8  : Inst<"svmls[_single]_za32[_{d}]_vg4x4", "vmi4d", "c",  "aarch64_sme_smls_za32_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVMLSL_SINGLE_VG4x4_U8  : Inst<"svmls[_single]_za32[_{d}]_vg4x4", "vmi4d", "Uc", "aarch64_sme_umls_za32_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;

  let TargetGuard = "sme-i16i64" in {
    def SVMLSL_SINGLE_VG4x1_S16 : Inst<"svmls[_single]_za64[_{d}]_vg4x1", "vmidd", "s",  "aarch64_sme_smls_za64_single_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>]>;
    def SVMLSL_SINGLE_VG4x1_U16 : Inst<"svmls[_single]_za64[_{d}]_vg4x1", "vmidd", "Us", "aarch64_sme_umls_za64_single_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>]>;
    def SVMLSL_SINGLE_VG4x2_S16 : Inst<"svmls[_single]_za64[_{d}]_vg4x2", "vmi2d", "s",  "aarch64_sme_smls_za64_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLSL_SINGLE_VG4x2_U16 : Inst<"svmls[_single]_za64[_{d}]_vg4x2", "vmi2d", "Us", "aarch64_sme_umls_za64_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLSL_SINGLE_VG4x4_S16 : Inst<"svmls[_single]_za64[_{d}]_vg4x4", "vmi4d", "s",  "aarch64_sme_smls_za64_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
    def SVMLSL_SINGLE_VG4x4_U16 : Inst<"svmls[_single]_za64[_{d}]_vg4x4", "vmi4d", "Us", "aarch64_sme_umls_za64_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  }

  // INDEXED MLAL
  def SVMLAL_LANE_VG2x1_F16 : Inst<"svmla_lane_za32[_{d}]_vg2x1", "vmiddi", "bh", "aarch64_sme_fmlal_lane_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLAL_LANE_VG2x2_F16 : Inst<"svmla_lane_za32[_{d}]_vg2x2", "vmi2di", "bh", "aarch64_sme_fmlal_lane_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLAL_LANE_VG2x4_F16 : Inst<"svmla_lane_za32[_{d}]_vg2x4", "vmi4di", "bh", "aarch64_sme_fmlal_lane_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLAL_LANE_VG2x1_S16 : Inst<"svmla_lane_za32[_{d}]_vg2x1", "vmiddi", "s",  "aarch64_sme_smlal_lane_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLAL_LANE_VG2x2_S16 : Inst<"svmla_lane_za32[_{d}]_vg2x2", "vmi2di", "s",  "aarch64_sme_smlal_lane_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLAL_LANE_VG2x4_S16 : Inst<"svmla_lane_za32[_{d}]_vg2x4", "vmi4di", "s",  "aarch64_sme_smlal_lane_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLAL_LANE_VG2x1_U16 : Inst<"svmla_lane_za32[_{d}]_vg2x1", "vmiddi", "Us", "aarch64_sme_umlal_lane_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLAL_LANE_VG2x2_U16 : Inst<"svmla_lane_za32[_{d}]_vg2x2", "vmi2di", "Us", "aarch64_sme_umlal_lane_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLAL_LANE_VG2x4_U16 : Inst<"svmla_lane_za32[_{d}]_vg2x4", "vmi4di", "Us", "aarch64_sme_umlal_lane_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;

  def SVMLAL_LANE_VG4x1_S8 : Inst<"svmla_lane_za32[_{d}]_vg4x1", "vmiddi", "c",  "aarch64_sme_smla_za32_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLAL_LANE_VG4x1_U8 : Inst<"svmla_lane_za32[_{d}]_vg4x1", "vmiddi", "Uc", "aarch64_sme_umla_za32_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLAL_LANE_VG4x2_S8 : Inst<"svmla_lane_za32[_{d}]_vg4x2", "vmi2di", "c",  "aarch64_sme_smla_za32_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLAL_LANE_VG4x2_U8 : Inst<"svmla_lane_za32[_{d}]_vg4x2", "vmi2di", "Uc", "aarch64_sme_umla_za32_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLAL_LANE_VG4x4_S8 : Inst<"svmla_lane_za32[_{d}]_vg4x4", "vmi4di", "c",  "aarch64_sme_smla_za32_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLAL_LANE_VG4x4_U8 : Inst<"svmla_lane_za32[_{d}]_vg4x4", "vmi4di", "Uc", "aarch64_sme_umla_za32_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;

  let TargetGuard = "sme-i16i64" in {
    def SVMLAL_LANE_VG4x1_S16 : Inst<"svmla_lane_za64[_{d}]_vg4x1", "vmiddi", "s",  "aarch64_sme_smla_za64_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLAL_LANE_VG4x1_U16 : Inst<"svmla_lane_za64[_{d}]_vg4x1", "vmiddi", "Us", "aarch64_sme_umla_za64_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLAL_LANE_VG4x2_S16 : Inst<"svmla_lane_za64[_{d}]_vg4x2", "vmi2di", "s",  "aarch64_sme_smla_za64_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLAL_LANE_VG4x2_U16 : Inst<"svmla_lane_za64[_{d}]_vg4x2", "vmi2di", "Us", "aarch64_sme_umla_za64_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLAL_LANE_VG4x4_S16 : Inst<"svmla_lane_za64[_{d}]_vg4x4", "vmi4di", "s",  "aarch64_sme_smla_za64_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLAL_LANE_VG4x4_U16 : Inst<"svmla_lane_za64[_{d}]_vg4x4", "vmi4di", "Us", "aarch64_sme_umla_za64_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
  }

  // INDEXED MLSL
  def SVMLSL_LANE_VG2x1_F16 : Inst<"svmls_lane_za32[_{d}]_vg2x1", "vmiddi", "bh", "aarch64_sme_fmlsl_lane_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLSL_LANE_VG2x2_F16 : Inst<"svmls_lane_za32[_{d}]_vg2x2", "vmi2di", "bh", "aarch64_sme_fmlsl_lane_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLSL_LANE_VG2x4_F16 : Inst<"svmls_lane_za32[_{d}]_vg2x4", "vmi4di", "bh", "aarch64_sme_fmlsl_lane_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLSL_LANE_VG2x1_S16 : Inst<"svmls_lane_za32[_{d}]_vg2x1", "vmiddi", "s",  "aarch64_sme_smlsl_lane_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLSL_LANE_VG2x2_S16 : Inst<"svmls_lane_za32[_{d}]_vg2x2", "vmi2di", "s",  "aarch64_sme_smlsl_lane_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLSL_LANE_VG2x4_S16 : Inst<"svmls_lane_za32[_{d}]_vg2x4", "vmi4di", "s",  "aarch64_sme_smlsl_lane_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLSL_LANE_VG2x1_U16 : Inst<"svmls_lane_za32[_{d}]_vg2x1", "vmiddi", "Us", "aarch64_sme_umlsl_lane_vg2x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_14_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLSL_LANE_VG2x2_U16 : Inst<"svmls_lane_za32[_{d}]_vg2x2", "vmi2di", "Us", "aarch64_sme_umlsl_lane_vg2x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;
  def SVMLSL_LANE_VG2x4_U16 : Inst<"svmls_lane_za32[_{d}]_vg2x4", "vmi4di", "Us", "aarch64_sme_umlsl_lane_vg2x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_6_Mul2>, ImmCheck<4, ImmCheck0_7>]>;

  def SVMLSL_LANE_VG4x1_S8 : Inst<"svmls_lane_za32[_{d}]_vg4x1", "vmiddi", "c",  "aarch64_sme_smls_za32_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLSL_LANE_VG4x1_U8 : Inst<"svmls_lane_za32[_{d}]_vg4x1", "vmiddi", "Uc", "aarch64_sme_umls_za32_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLSL_LANE_VG4x2_S8 : Inst<"svmls_lane_za32[_{d}]_vg4x2", "vmi2di", "c",  "aarch64_sme_smls_za32_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLSL_LANE_VG4x2_U8 : Inst<"svmls_lane_za32[_{d}]_vg4x2", "vmi2di", "Uc", "aarch64_sme_umls_za32_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLSL_LANE_VG4x4_S8 : Inst<"svmls_lane_za32[_{d}]_vg4x4", "vmi4di", "c",  "aarch64_sme_smls_za32_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVMLSL_LANE_VG4x4_U8 : Inst<"svmls_lane_za32[_{d}]_vg4x4", "vmi4di", "Uc", "aarch64_sme_umls_za32_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;

  let TargetGuard = "sme-i16i64" in {
    def SVMLSL_LANE_VG4x1_S16 : Inst<"svmls_lane_za64[_{d}]_vg4x1", "vmiddi", "s",  "aarch64_sme_smls_za64_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLSL_LANE_VG4x1_U16 : Inst<"svmls_lane_za64[_{d}]_vg4x1", "vmiddi", "Us", "aarch64_sme_umls_za64_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLSL_LANE_VG4x2_S16 : Inst<"svmls_lane_za64[_{d}]_vg4x2", "vmi2di", "s",  "aarch64_sme_smls_za64_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLSL_LANE_VG4x2_U16 : Inst<"svmls_lane_za64[_{d}]_vg4x2", "vmi2di", "Us", "aarch64_sme_umls_za64_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLSL_LANE_VG4x4_S16 : Inst<"svmls_lane_za64[_{d}]_vg4x4", "vmi4di", "s",  "aarch64_sme_smls_za64_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
    def SVMLSL_LANE_VG4x4_U16 : Inst<"svmls_lane_za64[_{d}]_vg4x4", "vmi4di", "Us", "aarch64_sme_umls_za64_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_7>]>;
  }

  // SINGLE SUMLALL
  def SVSUMLALL_SINGLE_VG4x2 : Inst<"svsumla[_single]_za32[_u8]_vg4x2", "vmi2.xd", "Uc", "aarch64_sme_sumla_za32_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVSUMLALL_SINGLE_VG4x4 : Inst<"svsumla[_single]_za32[_u8]_vg4x4", "vmi4.xd", "Uc", "aarch64_sme_sumla_za32_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;

  // INDEXED SUMLALL
  def SVSUMLALL_LANE_VG4x1 : Inst<"svsumla_lane_za32[_{d}]_vg4x1", "vmidui", "c", "aarch64_sme_sumla_za32_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVSUMLALL_LANE_VG4x2 : Inst<"svsumla_lane_za32[_{d}]_vg4x2", "vmi2ui", "c", "aarch64_sme_sumla_za32_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVSUMLALL_LANE_VG4x4 : Inst<"svsumla_lane_za32[_{d}]_vg4x4", "vmi4ui", "c", "aarch64_sme_sumla_za32_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;

  // SINGLE USMLALL
  def SVUSMLALL_SINGLE_VG4x1 : Inst<"svusmla[_single]_za32[_{d}]_vg4x1", "vmiud", "c", "aarch64_sme_usmla_za32_single_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>]>;
  def SVUSMLALL_SINGLE_VG4x2 : Inst<"svusmla[_single]_za32[_{d}]_vg4x2", "vmi2.ud", "c", "aarch64_sme_usmla_za32_single_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVUSMLALL_SINGLE_VG4x4 : Inst<"svusmla[_single]_za32[_{d}]_vg4x4", "vmi4.ud", "c", "aarch64_sme_usmla_za32_single_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;

  // MULTI USMLALL
  def SVUSMLALL_MULTI_VG4x2 : Inst<"svusmla_za32[_{d}]_vg4x2", "vmi2.u2", "c", "aarch64_sme_usmla_za32_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;
  def SVUSMLALL_MULTI_VG4x4 : Inst<"svusmla_za32[_{d}]_vg4x4", "vmi4.u4", "c", "aarch64_sme_usmla_za32_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>]>;

  // INDEXED USMLALL
  def SVUSMLALL_LANE_VG4x1 : Inst<"svusmla_lane_za32[_{d}]_vg4x1", "vmidxi", "Uc", "aarch64_sme_usmla_za32_lane_vg4x1", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_12_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVUSMLALL_LANE_VG4x2 : Inst<"svusmla_lane_za32[_{d}]_vg4x2", "vmi2xi", "Uc", "aarch64_sme_usmla_za32_lane_vg4x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;
  def SVUSMLALL_LANE_VG4x4 : Inst<"svusmla_lane_za32[_{d}]_vg4x4", "vmi4xi", "Uc", "aarch64_sme_usmla_za32_lane_vg4x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_4_Mul4>, ImmCheck<4, ImmCheck0_15>]>;

  // VERTICAL DOT-PRODUCT
  def SVVDOT_LANE_ZA32_VG1x2_S : Inst<"svvdot_lane_za32[_{d}]_vg1x2", "vmi2di", "s",  "aarch64_sme_svdot_lane_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVVDOT_LANE_ZA32_VG1x4_S : Inst<"svvdot_lane_za32[_{d}]_vg1x4", "vmi4di", "c",  "aarch64_sme_svdot_lane_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVVDOT_LANE_ZA32_VG1x2_U : Inst<"svvdot_lane_za32[_{d}]_vg1x2", "vmi2di", "Us", "aarch64_sme_uvdot_lane_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVVDOT_LANE_ZA32_VG1x4_U : Inst<"svvdot_lane_za32[_{d}]_vg1x4", "vmi4di", "Uc", "aarch64_sme_uvdot_lane_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVVDOT_LANE_ZA32_VG1x2_F : Inst<"svvdot_lane_za32[_{d}]_vg1x2", "vmi2di", "hb", "aarch64_sme_fvdot_lane_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVSUVDOT_LANE_ZA32_VG1x4 : Inst<"svsuvdot_lane_za32[_{d}]_vg1x4", "vmi4di", "c",  "aarch64_sme_suvdot_lane_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVUSVDOT_LANE_ZA32_VG1x4 : Inst<"svusvdot_lane_za32[_{d}]_vg1x4", "vmi4di", "Uc", "aarch64_sme_usvdot_lane_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;

  let TargetGuard = "sme-i16i64" in {
  def SVVDOT_LANE_ZA64_VG1x4_S : Inst<"svvdot_lane_za64[_{d}]_vg1x4", "vmi4di", "s",  "aarch64_sme_svdot_lane_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
  def SVVDOT_LANE_ZA64_VG1x4_U : Inst<"svvdot_lane_za64[_{d}]_vg1x4", "vmi4di", "Us", "aarch64_sme_uvdot_lane_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
  }

  // Multi-vector signed & unsigned integer dot-product
  def SVDOT_SINGLE_ZA32_VG1x2_S : Inst<"svdot[_single]_za32[_{d}]_vg1x2", "vmi2d", "cs", "aarch64_sme_sdot_single_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_SINGLE_ZA32_VG1x4_S : Inst<"svdot[_single]_za32[_{d}]_vg1x4", "vmi4d", "cs", "aarch64_sme_sdot_single_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_SINGLE_ZA32_VG1x2_U : Inst<"svdot[_single]_za32[_{d}]_vg1x2", "vmi2d", "UcUs", "aarch64_sme_udot_single_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_SINGLE_ZA32_VG1x4_U : Inst<"svdot[_single]_za32[_{d}]_vg1x4", "vmi4d", "UcUs", "aarch64_sme_udot_single_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA32_VG1x2_S  : Inst<"svdot_za32[_{d}]_vg1x2", "vmi22", "cs", "aarch64_sme_sdot_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA32_VG1x4_S  : Inst<"svdot_za32[_{d}]_vg1x4", "vmi44", "cs", "aarch64_sme_sdot_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA32_VG1x2_U  : Inst<"svdot_za32[_{d}]_vg1x2", "vmi22", "UcUs", "aarch64_sme_udot_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA32_VG1x4_U  : Inst<"svdot_za32[_{d}]_vg1x4", "vmi44", "UcUs", "aarch64_sme_udot_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_LANE_ZA32_VG1x2_S   : Inst<"svdot_lane_za32[_{d}]_vg1x2", "vmi2di", "cs", "aarch64_sme_sdot_lane_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVDOT_LANE_ZA32_VG1x4_S   : Inst<"svdot_lane_za32[_{d}]_vg1x4", "vmi4di", "cs", "aarch64_sme_sdot_lane_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVDOT_LANE_ZA32_VG1x2_U   : Inst<"svdot_lane_za32[_{d}]_vg1x2", "vmi2di", "UcUs", "aarch64_sme_udot_lane_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVDOT_LANE_ZA32_VG1x4_U   : Inst<"svdot_lane_za32[_{d}]_vg1x4", "vmi4di", "UcUs", "aarch64_sme_udot_lane_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;

  def SVUSDOT_SINGLE_ZA32_VG1x2 : Inst<"svusdot[_single]_za32[_{d}]_vg1x2", "vmi2.dx", "Uc", "aarch64_sme_usdot_single_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVUSDOT_SINGLE_ZA32_VG1x4 : Inst<"svusdot[_single]_za32[_{d}]_vg1x4", "vmi4.dx", "Uc", "aarch64_sme_usdot_single_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVUSDOT_MULTI_ZA32_VG1x2  : Inst<"svusdot_za32[_{d}]_vg1x2", "vmi2.d2.x", "Uc", "aarch64_sme_usdot_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVUSDOT_MULTI_ZA32_VG1x4  : Inst<"svusdot_za32[_{d}]_vg1x4", "vmi4.d4.x", "Uc", "aarch64_sme_usdot_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVUSDOT_LANE_ZA32_VG1x2   : Inst<"svusdot_lane_za32[_{d}]_vg1x2", "vmi2.dxi", "Uc", "aarch64_sme_usdot_lane_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVUSDOT_LANE_ZA32_VG1x4   : Inst<"svusdot_lane_za32[_{d}]_vg1x4", "vmi4.dxi", "Uc", "aarch64_sme_usdot_lane_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;

  def SVSUDOT_SINGLE_ZA32_VG1x2 : Inst<"svsudot[_single]_za32[_{d}]_vg1x2", "vmi2.du", "c", "aarch64_sme_sudot_single_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVSUDOT_SINGLE_ZA32_VG1x4 : Inst<"svsudot[_single]_za32[_{d}]_vg1x4", "vmi4.du", "c", "aarch64_sme_sudot_single_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVSUDOT_LANE_ZA32_VG1x2   : Inst<"svsudot_lane_za32[_{d}]_vg1x2", "vmi2.dui", "c", "aarch64_sme_sudot_lane_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVSUDOT_LANE_ZA32_VG1x4   : Inst<"svsudot_lane_za32[_{d}]_vg1x4", "vmi4.dui", "c", "aarch64_sme_sudot_lane_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;

  let TargetGuard = "sme-i16i64" in {
  def SVDOT_SINGLE_ZA64_VG1x2_S16 : Inst<"svdot[_single]_za64[_{d}]_vg1x2", "vmi2d", "s", "aarch64_sme_sdot_single_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_SINGLE_ZA64_VG1x4_S16 : Inst<"svdot[_single]_za64[_{d}]_vg1x4", "vmi4d", "s", "aarch64_sme_sdot_single_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_SINGLE_ZA64_VG1x2_U16 : Inst<"svdot[_single]_za64[_{d}]_vg1x2", "vmi2d", "Us", "aarch64_sme_udot_single_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_SINGLE_ZA64_VG1x4_U16 : Inst<"svdot[_single]_za64[_{d}]_vg1x4", "vmi4d", "Us", "aarch64_sme_udot_single_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA64_VG1x2_S16  : Inst<"svdot_za64[_{d}]_vg1x2", "vmi22", "s", "aarch64_sme_sdot_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA64_VG1x4_S16  : Inst<"svdot_za64[_{d}]_vg1x4", "vmi44", "s", "aarch64_sme_sdot_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA64_VG1x2_U16  : Inst<"svdot_za64[_{d}]_vg1x2", "vmi22", "Us", "aarch64_sme_udot_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA64_VG1x4_U16  : Inst<"svdot_za64[_{d}]_vg1x4", "vmi44", "Us", "aarch64_sme_udot_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_LANE_ZA64_VG1x2_S16   : Inst<"svdot_lane_za64[_{d}]_vg1x2", "vmi2di", "s", "aarch64_sme_sdot_lane_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
  def SVDOT_LANE_ZA64_VG1x4_S16   : Inst<"svdot_lane_za64[_{d}]_vg1x4", "vmi4di", "s", "aarch64_sme_sdot_lane_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
  def SVDOT_LANE_ZA64_VG1x2_U16   : Inst<"svdot_lane_za64[_{d}]_vg1x2", "vmi2di", "Us", "aarch64_sme_udot_lane_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
  def SVDOT_LANE_ZA64_VG1x4_U16   : Inst<"svdot_lane_za64[_{d}]_vg1x4", "vmi4di", "Us", "aarch64_sme_udot_lane_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_1>]>;
  }

  // Multi-vector half-precision/BFloat16 floating-point dot-product
  def SVDOT_SINGLE_ZA32_VG1x2_F16 : Inst<"svdot[_single]_za32[_{d}]_vg1x2", "vmi2d", "bh", "aarch64_sme_fdot_single_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_SINGLE_ZA32_VG1x4_F16 : Inst<"svdot[_single]_za32[_{d}]_vg1x4", "vmi4d", "bh", "aarch64_sme_fdot_single_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA32_VG1x2_F16  : Inst<"svdot_za32[_{d}]_vg1x2", "vmi22", "bh", "aarch64_sme_fdot_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_MULTI_ZA32_VG1x4_F16  : Inst<"svdot_za32[_{d}]_vg1x4", "vmi44", "bh", "aarch64_sme_fdot_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVDOT_LANE_ZA32_VG1x2_F16   : Inst<"svdot_lane_za32[_{d}]_vg1x2", "vmi2di", "bh", "aarch64_sme_fdot_lane_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
  def SVDOT_LANE_ZA32_VG1x4_F16   : Inst<"svdot_lane_za32[_{d}]_vg1x4", "vmi4di", "bh", "aarch64_sme_fdot_lane_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>, ImmCheck<4, ImmCheck0_3>]>;
}

// == ADD /SUB ==
let TargetGuard = "sme2" in {
  def SVADD_WRITE_SINGLE_ZA32_VG1X2_I32 : Inst<"svadd_write[_single]_za32[_{d}]_vg1x2", "vmi2d", "iUi", "aarch64_sme_add_write_single_za_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVADD_WRITE_SINGLE_ZA32_VG1X4_I32 : Inst<"svadd_write[_single]_za32[_{d}]_vg1x4", "vmi4d", "iUi", "aarch64_sme_add_write_single_za_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  def SVSUB_WRITE_SINGLE_ZA32_VG1X2_I32 : Inst<"svsub_write[_single]_za32[_{d}]_vg1x2", "vmi2d", "Ui", "aarch64_sme_sub_write_single_za_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVSUB_WRITE_SINGLE_ZA32_VG1X4_I32 : Inst<"svsub_write[_single]_za32[_{d}]_vg1x4", "vmi4d", "Ui", "aarch64_sme_sub_write_single_za_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  def SVADD_WRITE_ZA32_VG1x2_I32 : Inst<"svadd_write_za32[_{d}]_vg1x2", "vmi22", "iUi", "aarch64_sme_add_write_za_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVADD_WRITE_ZA32_VG1x4_I32 : Inst<"svadd_write_za32[_{d}]_vg1x4", "vmi44", "iUi", "aarch64_sme_add_write_za_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  def SVSUB_WRITE_ZA32_VG1x2_I32 : Inst<"svsub_write_za32[_{d}]_vg1x2", "vmi22", "Ui", "aarch64_sme_sub_write_za_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVSUB_WRITE_ZA32_VG1x4_I32 : Inst<"svsub_write_za32[_{d}]_vg1x4", "vmi44", "Ui", "aarch64_sme_sub_write_za_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  def SVADD_ZA32_VG1x2_I32 : Inst<"svadd_za32[_{d}]_vg1x2", "vmi2", "iUif", "aarch64_sme_add_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVADD_ZA32_VG1X4_I32 : Inst<"svadd_za32[_{d}]_vg1x4", "vmi4", "iUif", "aarch64_sme_add_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  def SVSUB_ZA32_VG1X2_I32 : Inst<"svsub_za32[_{d}]_vg1x2", "vmi2", "Uif", "aarch64_sme_sub_za32_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVSUB_ZA32_VG1X4_I32 : Inst<"svsub_za32[_{d}]_vg1x4", "vmi4", "Uif", "aarch64_sme_sub_za32_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

  let TargetGuard = "sme-i16i64" in {
    def SVADD_WRITE_SINGLE_ZA64_VG1X2_I64 : Inst<"svadd_write[_single]_za64[_{d}]_vg1x2", "vmi2d", "lUl", "aarch64_sme_add_write_single_za_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
    def SVADD_WRITE_SINGLE_ZA64_VG1X4_I64 : Inst<"svadd_write[_single]_za64[_{d}]_vg1x4", "vmi4d", "lUl", "aarch64_sme_add_write_single_za_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

    def SVSUB_WRITE_SINGLE_ZA64_VG1X2_I64 : Inst<"svsub_write[_single]_za64[_{d}]_vg1x2", "vmi2d", "Ul", "aarch64_sme_sub_write_single_za_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
    def SVSUB_WRITE_SINGLE_ZA64_VG1X4_I64 : Inst<"svsub_write[_single]_za64[_{d}]_vg1x4", "vmi4d", "Ul", "aarch64_sme_sub_write_single_za_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

    def SVADD_WRITE_ZA64_VG1x2_I64 : Inst<"svadd_write_za64[_{d}]_vg1x2", "vmi22", "lUl", "aarch64_sme_add_write_za_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
    def SVADD_WRITE_ZA64_VG1x4_I64 : Inst<"svadd_write_za64[_{d}]_vg1x4", "vmi44", "lUl", "aarch64_sme_add_write_za_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

    def SVSUB_WRITE_ZA64_VG1x2_I64 : Inst<"svsub_write_za64[_{d}]_vg1x2", "vmi22", "Ul", "aarch64_sme_sub_write_za_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
    def SVSUB_WRITE_ZA64_VG1x4_I64 : Inst<"svsub_write_za64[_{d}]_vg1x4", "vmi44", "Ul", "aarch64_sme_sub_write_za_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

    def SVADD_ZA64_VG1X2_I64 : Inst<"svadd_za64[_{d}]_vg1x2", "vmi2", "lUl", "aarch64_sme_add_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
    def SVADD_ZA64_VG1X4_I64 : Inst<"svadd_za64[_{d}]_vg1x4", "vmi4", "lUl", "aarch64_sme_add_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

    def SVSUB_ZA64_VG1X2_I64 : Inst<"svsub_za64[_{d}]_vg1x2", "vmi2", "Ul", "aarch64_sme_sub_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
    def SVSUB_ZA64_VG1X4_I64 : Inst<"svsub_za64[_{d}]_vg1x4", "vmi4", "Ul", "aarch64_sme_sub_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  }

  let TargetGuard = "sme-f64f64" in {
    def SVADD_ZA64_VG1X2_F64 : Inst<"svadd_za64[_{d}]_vg1x2", "vmi2", "d", "aarch64_sme_add_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
    def SVADD_ZA64_VG1X4_F64 : Inst<"svadd_za64[_{d}]_vg1x4", "vmi4", "d", "aarch64_sme_add_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;

    def SVSUB_ZA64_VG1X2_F64 : Inst<"svsub_za64[_{d}]_vg1x2", "vmi2", "d", "aarch64_sme_sub_za64_vg1x2", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
    def SVSUB_ZA64_VG1X4_F64 : Inst<"svsub_za64[_{d}]_vg1x4", "vmi4", "d", "aarch64_sme_sub_za64_vg1x4", [IsStreaming, IsSharedZA, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  }
}
//
// 2 and 4 vector-group read/write intrinsics.
//

multiclass WriteHV_VG<string n, string t, string i, list<ImmCheck> checkvg2, list<ImmCheck> checkvg4> {
  def NAME # _VG2_H : Inst<"svwrite_hor_" # n # "_vg2", "vimi2", t, i # "_hor_vg2", [IsSharedZA, IsStreaming, IsTileBaseOffsetIntr], checkvg2>;
  def NAME # _VG2_V : Inst<"svwrite_ver_" # n # "_vg2", "vimi2", t, i # "_ver_vg2",  [IsSharedZA, IsStreaming, IsTileBaseOffsetIntr], checkvg2>;
  def NAME # _VG4_H : Inst<"svwrite_hor_" # n # "_vg4", "vimi4", t, i # "_hor_vg4", [IsSharedZA, IsStreaming, IsTileBaseOffsetIntr], checkvg4>;
  def NAME # _VG4_V : Inst<"svwrite_ver_" # n # "_vg4", "vimi4", t, i # "_ver_vg4",  [IsSharedZA, IsStreaming, IsTileBaseOffsetIntr], checkvg4>;
}

let TargetGuard = "sme2" in {
  defm SVWRITE_ZA8  : WriteHV_VG<"za8[_{d}]",  "cUc",   "aarch64_sme_write", [ImmCheck<0, ImmCheck0_0>,   ImmCheck<2, ImmCheck0_14_Mul2>], [ImmCheck<0, ImmCheck0_0>,   ImmCheck<2, ImmCheck0_12_Mul4>]>;
  defm SVWRITE_ZA16 : WriteHV_VG<"za16[_{d}]", "sUshb", "aarch64_sme_write", [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_6_Mul2>],  [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_4_Mul4>]>;
  defm SVWRITE_ZA32 : WriteHV_VG<"za32[_{d}]", "iUif",  "aarch64_sme_write", [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_2_Mul2>],  [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_0>]>;
  defm SVWRITE_ZA64 : WriteHV_VG<"za64[_{d}]", "lUld",  "aarch64_sme_write", [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_0>],         [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_0>]>;
}

multiclass ReadHV_VG<string n, string t, string i, list<ImmCheck> checkvg2, list<ImmCheck> checkvg4> {
  def NAME # _VG2_H : Inst<"svread_hor_" # n # "_vg2", "2imi", t, i # "_hor_vg2", [IsSharedZA, IsPreservesZA, IsStreaming, IsTileBaseOffsetIntr], checkvg2>;
  def NAME # _VG2_V : Inst<"svread_ver_" # n # "_vg2", "2imi", t, i # "_ver_vg2",  [IsSharedZA, IsPreservesZA, IsStreaming, IsTileBaseOffsetIntr], checkvg2>;
  def NAME # _VG4_H : Inst<"svread_hor_" # n # "_vg4", "4imi", t, i # "_hor_vg4", [IsSharedZA, IsPreservesZA, IsStreaming, IsTileBaseOffsetIntr], checkvg4>;
  def NAME # _VG4_V : Inst<"svread_ver_" # n # "_vg4", "4imi", t, i # "_ver_vg4",  [IsSharedZA, IsPreservesZA, IsStreaming, IsTileBaseOffsetIntr], checkvg4>;
}

let TargetGuard = "sme2" in {
  defm SVREAD_ZA8  : ReadHV_VG<"za8_{d}",  "cUc",   "aarch64_sme_read", [ImmCheck<0, ImmCheck0_0>,   ImmCheck<2, ImmCheck0_14_Mul2>], [ImmCheck<0, ImmCheck0_0>,   ImmCheck<2, ImmCheck0_12_Mul4>]>;
  defm SVREAD_ZA16 : ReadHV_VG<"za16_{d}", "sUshb", "aarch64_sme_read", [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_6_Mul2>],  [ImmCheck<0, ImmCheck0_1>, ImmCheck<2, ImmCheck0_4_Mul4>]>;
  defm SVREAD_ZA32 : ReadHV_VG<"za32_{d}", "iUif",  "aarch64_sme_read", [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_2_Mul2>],  [ImmCheck<0, ImmCheck0_3>, ImmCheck<2, ImmCheck0_0>]>;
  defm SVREAD_ZA64 : ReadHV_VG<"za64_{d}", "lUld",  "aarch64_sme_read", [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_0>],         [ImmCheck<0, ImmCheck0_7>, ImmCheck<2, ImmCheck0_0>]>;
}


//
// Single vector-group read/write intrinsics.
//

let TargetGuard = "sme2" in {
  def SVWRITE_ZA64_VG1x2 : Inst<"svwrite_za64[_{d}]_vg1x2", "vmi2", "lUld", "aarch64_sme_write_vg1x2", [IsSharedZA, IsStreaming, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVWRITE_ZA64_VG1x4 : Inst<"svwrite_za64[_{d}]_vg1x4", "vmi4", "lUld", "aarch64_sme_write_vg1x4", [IsSharedZA, IsStreaming, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVREAD_ZA64_VG1x2  : Inst<"svread_za64_{d}_vg1x2",  "2mi",  "lUld", "aarch64_sme_read_vg1x2",  [IsSharedZA, IsPreservesZA, IsStreaming, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
  def SVREAD_ZA64_VG1x4  : Inst<"svread_za64_{d}_vg1x4",  "4mi",  "lUld", "aarch64_sme_read_vg1x4",  [IsSharedZA, IsPreservesZA, IsStreaming, IsZASliceBaseOffsetIntr], [ImmCheck<1, ImmCheck0_7>]>;
}

//
// Outer produce and accumulate/subtract
//

let TargetGuard = "sme2" in {
  def SVSMOPA  : Inst<"svmopa_za32[_{d}]_m", "viPPdd", "s",  "aarch64_sme_smopa_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;
  def SVUSMOPA : Inst<"svmopa_za32[_{d}]_m", "viPPdd", "Us", "aarch64_sme_umopa_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;

  def SVSMOPS  : Inst<"svmops_za32[_{d}]_m", "viPPdd", "s",  "aarch64_sme_smops_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;
  def SVUSMOPS : Inst<"svmops_za32[_{d}]_m", "viPPdd", "Us", "aarch64_sme_umops_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;

  def SVBMOPA : Inst<"svbmopa_za32[_{d}]_m", "viPPdd", "Ui", "aarch64_sme_bmopa_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;

  def SVBMOPS : Inst<"svbmops_za32[_{d}]_m", "viPPdd", "Ui", "aarch64_sme_bmops_za32", [IsSharedZA, IsStreaming], [ImmCheck<0, ImmCheck0_3>]>;
}

//
// Spill and fill of ZT0
//

let TargetGuard = "sme2" in {
  def SVLDR_ZT : Inst<"svldr_zt", "viQ", "", "aarch64_sme_ldr_zt", [IsOverloadNone, IsStreamingCompatible, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>]>;
  def SVSTR_ZT : Inst<"svstr_zt", "vi%", "", "aarch64_sme_str_zt", [IsOverloadNone, IsStreamingCompatible, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>]>;
}

//
// Zero ZT0
//

let TargetGuard = "sme2" in {
  def SVZERO_ZT : Inst<"svzero_zt", "vi", "", "aarch64_sme_zero_zt", [IsOverloadNone, IsStreamingCompatible, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>]>;
}

//
// lookup table expand one register
//

let TargetGuard = "sme2" in {
  def SVLUTI2_LANE_ZT : Inst<"svluti2_lane_zt[_{d}]", "didi", "UcUsUi", "aarch64_sme_luti2_lane_zt", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_15>]>;
  def SVLUTI4_LANE_ZT : Inst<"svluti4_lane_zt[_{d}]", "didi", "UcUsUi", "aarch64_sme_luti4_lane_zt", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_7>]>;
}

//
// lookup table expand two contiguous registers
//

let TargetGuard = "sme2" in {
  def SVLUTI2_LANE_ZT_X2 : Inst<"svluti2_lane_zt[_{d}]_x2", "2.didi", "UcUsUi", "aarch64_sme_luti2_lane_zt_x2", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_7>]>;
  def SVLUTI4_LANE_ZT_X2 : Inst<"svluti4_lane_zt[_{d}]_x2", "2.didi", "UcUsUi", "aarch64_sme_luti4_lane_zt_x2", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_3>]>;
}

//
// lookup table expand four contiguous registers
//

let TargetGuard = "sme2" in {
  def SVLUTI2_LANE_ZT_X4 : Inst<"svluti2_lane_zt[_{d}]_x4", "4.didi", "UcUsUi", "aarch64_sme_luti2_lane_zt_x4", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_3>]>;
  def SVLUTI4_LANE_ZT_X4 : Inst<"svluti4_lane_zt[_{d}]_x4", "4.didi", "UsUi", "aarch64_sme_luti4_lane_zt_x4", [IsStreaming, IsSharedZA, IsPreservesZA], [ImmCheck<0, ImmCheck0_0>, ImmCheck<2, ImmCheck0_1>]>;
}
