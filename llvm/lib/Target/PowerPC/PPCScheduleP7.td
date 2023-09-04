//===-- PPCScheduleP7.td - PPC P7 Scheduling Definitions ---*- tablegen -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file defines the SchedModel for the POWER7 processor.
//
//===----------------------------------------------------------------------===//

def P7Model : SchedMachineModel {
  let IssueWidth = 6;  // 4 (non-branch) instructions are dispatched per cycle.
                       // Note that the dispatch bundle size is 6 (including
                       // branches), but the total internal issue bandwidth per
                       // cycle (from all queues) is 8.

  let LoadLatency = 3; // Optimistic load latency assuming bypass.
                       // This is overriden by OperandCycles if the
                       // Itineraries are queried instead.
  let MispredictPenalty = 16;

  let MicroOpBufferSize = 44;

  // Try to make sure we have at least 10 dispatch groups in a loop.
  let LoopMicroOpBufferSize = 40;

  let CompleteModel = 0;

  let UnsupportedFeatures = [HasSPE, PrefixInstrs, MMA,
                             PairedVectorMemops, IsISA3_0,
                             PCRelativeMemops, IsISA3_1, IsISAFuture];
}

let SchedModel = P7Model in {
  def P7_LSU_FXU: ProcResource<4>;
  def P7_LSU: ProcResource<2> {
    let Super = P7_LSU_FXU;
  }
  def P7_FXU: ProcResource<2> {
    let Super = P7_LSU_FXU;
  }
  // Implemented as two 2-way SIMD operations for double- and single-precision.
  def P7_FPU: ProcResource<4>;
  // Scalar binary floating point instructions can only use two FPUs.
  def P7_ScalarFPU: ProcResource<2> {
    let Super = P7_FPU;
  }
  // Executing simple FX, complex FX, permute and 4-way SIMD single-precision FP ops
  def P7_VMX: ProcResource<2>;
  def P7_VPM: ProcResource<2> {
    let Super = P7_VMX;
    let BufferSize = 1;
  }
  def P7_DFU: ProcResource<1>;
  def P7_BRU: ProcResource<1>;
  def P7_CRU: ProcResource<1>;

  def P7_PORT_1 : ProcResource<2>;
  def P7_PORT_2 : ProcResource<2>;
  def P7_PORT_3 : ProcResource<2>;
  def P7_PORT_4 : ProcResource<1>;
  def P7_PORT_5 : ProcResource<1>;

  def P7_DISP_LS : SchedWriteRes<[P7_PORT_1]>;
  def P7_DISP_FX : SchedWriteRes<[P7_PORT_2]>;
  def P7_DISP_FP : SchedWriteRes<[P7_PORT_3]>;
  def P7_DISP_BR : SchedWriteRes<[P7_PORT_4]>;
  def P7_DISP_CR : SchedWriteRes<[P7_PORT_5]>;

  def P7_NONE_0C : SchedWriteRes<[]> { let Latency = 0; }
  def P7_BRU_1C : SchedWriteRes<[P7_BRU]>;
  def P7_BRU_3C : SchedWriteRes<[P7_BRU]> { let Latency = 3; }
  def P7_BRU_4C : SchedWriteRes<[P7_BRU]> { let Latency = 4; }
  def P7_CRU_3C : SchedWriteRes<[P7_CRU]> { let Latency = 3; }
  def P7_CRU_6C : SchedWriteRes<[P7_CRU]> { let Latency = 6; }
  def P7_LSU_1C : SchedWriteRes<[P7_LSU]> { let Latency = 1; }
  def P7_LSU_2C : SchedWriteRes<[P7_LSU]> { let Latency = 2; }
  def P7_LSU_3C : SchedWriteRes<[P7_LSU]> { let Latency = 3; }
  def P7_LSU_4C : SchedWriteRes<[P7_LSU]> { let Latency = 4; }
  def P7_FXU_1C : SchedWriteRes<[P7_FXU]> { let Latency = 1; }
  def P7_FXU_2C : SchedWriteRes<[P7_FXU]> { let Latency = 2; }
  def P7_FXU_3C : SchedWriteRes<[P7_FXU]> { let Latency = 3; }
  def P7_FXU_4C : SchedWriteRes<[P7_FXU]> { let Latency = 4; }
  def P7_FXU_5C : SchedWriteRes<[P7_FXU]> { let Latency = 5; }
  def P7_FXU_38C : SchedWriteRes<[P7_FXU]> { let Latency = 38; }
  def P7_FXU_69C : SchedWriteRes<[P7_FXU]> { let Latency = 69; }
  def P7_LSU_FXU_2C : SchedWriteRes<[P7_LSU_FXU]> { let Latency = 2; }
  def P7_FPU_6C : SchedWriteRes<[P7_FPU]> { let Latency = 6; }
  def P7_ScalarFPU_6C : SchedWriteRes<[P7_ScalarFPU]> { let Latency = 6; }
  def P7_ScalarFPU_8C : SchedWriteRes<[P7_ScalarFPU]> { let Latency = 8; }
  def P7_ScalarFPU_27C : SchedWriteRes<[P7_ScalarFPU]> { let Latency = 27; }
  def P7_ScalarFPU_32C : SchedWriteRes<[P7_ScalarFPU]> { let Latency = 32; }
  def P7_ScalarFPU_33C : SchedWriteRes<[P7_ScalarFPU]> { let Latency = 33; }
  def P7_ScalarFPU_44C : SchedWriteRes<[P7_ScalarFPU]> { let Latency = 44; }
  def P7_VMX_6C : SchedWriteRes<[P7_VMX]> { let Latency = 6; }
  def P7_VPM_6C : SchedWriteRes<[P7_VPM]> { let Latency = 6; }

  def : InstRW<[P7_NONE_0C], (instrs
    NOP, NOP_GT_PWR6, NOP_GT_PWR7
  )>;

  def : InstRW<[P7_BRU_1C, P7_DISP_BR], (instregex "^B(L)?(A)?(8)?(_NOP|_NOTOC)?(_TLS|_RM)?(_)?$")>;

  def : InstRW<[P7_BRU_3C, P7_DISP_BR], (instrs
    BDZLRLp, BDZLRm, BDZLRp, BDZLm, BDZLp, BDZm, BDZp,
    BDNZ, BDNZ8, BDNZA, BDNZAm, BDNZAp, BDNZL, BDNZLA, BDNZLAm, BDNZLAp, BDNZLR,
    BDNZLR8, BDNZLRL, BDNZLRLm, BDNZLRLp, BDNZLRm, BDNZLRp, BDNZLm, BDNZLp, BDNZm, BDNZp, BDZ,
    BDZ8, BDZA, BDZAm, BDZAp, BDZL, BDZLA, BDZLAm, BDZLAp, BDZLR, BDZLR8, BDZLRL, BDZLRLm,
    MFCTR, MFCTR8, MFLR, MFLR8,
    BLR, BLR8, BLRL, BCL, BCLR, BCLRL, BCLRLn, BCLRn, BCLalways, BCLn, BCTR, BCTR8, BCTRL, BCTRL8, BCTRL8_LDinto_toc,
    BCTRL8_LDinto_toc_RM, BCTRL8_RM, BCTRL_LWZinto_toc, BCTRL_LWZinto_toc_RM, BCTRL_RM, BCn, BC, BCC, BCCA,
    BCCCTR, BCCCTR8, BCCCTRL, BCCCTRL8, BCCL, BCCLA, BCCLR, BCCLRL, BCCTR, BCCTR8, BCCTR8n, BCCTRL, BCCTRL8,
    BCCTRL8n, BCCTRLn, BCCTRn, gBC, gBCA, gBCAat, gBCCTR, gBCCTRL, gBCL, gBCLA, gBCLAat, gBCLR, gBCLRL, gBCLat, gBCat
  )>;

  def : InstRW<[P7_BRU_4C], (instrs MTLR, MTLR8, MTCTR, MTCTR8, MTCTR8loop, MTCTRloop)>;

  def : InstRW<[P7_CRU_3C], (instrs MCRF)>;

  def : InstRW<[P7_CRU_6C, P7_DISP_CR], (instrs
    CMPD, CMPDI, CMPLD, CMPLDI, CMPLW, CMPLWI, CMPW, CMPWI, COPY,
    CR6SET, CR6UNSET, CRSET, CRUNSET,
    CRAND, CRANDC, CREQV, CRNAND, CRNOR, CRNOT, CROR, CRORC
  )>;

  def : InstRW<[P7_LSU_2C, P7_DISP_LS], (instrs LHBRX, LHBRX8, LWBRX, LWBRX8)>;

  def : InstRW<[P7_LSU_3C], (instrs MFSR, MFSRIN)>;

  def : InstRW<[P7_LSU_4C, P7_FXU_4C, P7_DISP_LS], (instrs
    LBARX, LBZCIX, LDBRX, LDCIX, LFD, LFDU, LFDUX, LFDX, LFIWAX, LFIWZX,
    LFS, LFSU, LFSUX, LFSX, LHARX, LHZCIX, LSWI, LVEBX, LVEHX, LVEWX, LVSL,
    LVSR, LVX, LVXL, LWZCIX, LXSDX, LXVD2X, LXVDSX, LXVW4X,
    STFD, STFDU, STFDUX, STFDX, STFIWX, STFS, STFSU, STFSUX, STFSX,
    STHCIX, STSWI, STVEBX, STVEHX, STVEWX, STVX, STVXL, STWCIX, STXSDX, STXVD2X, STXVW4X,
    LHA, LHA8, LHAX, LHAX8, LWA, LWAX, LWAX_32, LWA_32, LHAU, LHAU8, LHAUX, LHAUX8, LWAUX
  )>;

  def : InstRW<[P7_LSU_2C, P7_FXU_2C, P7_DISP_LS], (instrs
    LWZU, LWZU8, LHZU, LHZU8, LBZU, LBZU8, LDU,
    LWZUX, LWZUX8, LHZUX, LHZUX8, LBZUX, LBZUX8, LDUX
  )>;

  def : InstRW<[P7_LSU_1C, P7_FXU_1C, P7_DISP_LS], (instrs
    STB, STB8, STH, STH8, STW, STW8, STD,
    STBX, STBX8, STHX, STHX8, STWX, STWX8, STDX,
    STHBRX, STWBRX, STMW, STWCX, STDCX, STDU, STHU, STHU8, STBU, STBU8, STWU, STWU8,
    STDUX, STWUX, STWUX8, STHUX, STHUX8, STBUX, STBUX8
  )>;

  def : InstRW<[P7_LSU_1C, P7_DISP_LS], (instrs LMW, LWARX, LWARXL, LDARX, LDARXL)>;

  def : InstRW<[P7_LSU_FXU_2C, P7_DISP_FX], (instrs
    (instregex "^(ADD|L)I(S)?(8)?$"),
    (instregex "^(ADD|SUBF)(4|8)(TLS)?(_)?(_rec)?$"),
    (instregex "^(X)?ORI(S)?(8)?$"),
    (instregex "^(X)OR(8)?(_rec)?$"),
    ADDIC, ADDIC8, SUBFIC, SUBFIC8, SUBFZE, SUBFZE8,
    ADDE, ADDE8, ADDME, ADDME8, SUBFME, SUBFME8,
    NEG, NEG8, NEG8_rec, NEG_rec, NEG8O, NEGO,
    ANDI_rec, ANDIS_rec, AND, AND8, AND_rec, AND8_rec,
    NAND, NAND8, NAND_rec, NAND8_rec, NOR, NOR8, NOR_rec, NOR8_rec,
    EQV, EQV8, EQV_rec, EQV8_rec, ANDC, ANDC8, ANDC_rec, ANDC8_rec,
    ORC, ORC8, ORC_rec, ORC8_rec
  )>;

  def : InstRW<[P7_FXU_2C, P7_DISP_FX], (instrs
    EXTSB8_32_64, EXTSB8_rec, EXTSH8_32_64, EXTSH8_rec, EXTSW_32, EXTSW_32_64, EXTSW_32_64_rec,
    POPCNTB, POPCNTB8, POPCNTD, POPCNTW,
    ADDPCIS, ANDI8_rec, ANDIS8_rec,
    SUBFUS, SUBFUS_rec,
    ADD4O, ADD8O, ADDC, ADDC8, SUBFO, SUBF8O, SUBFC, SUBFC8,
    ADDIC_rec, ADDE8_rec, ADDE_rec, SUBFE8_rec, SUBFE_rec,
    ADDME8_rec, ADDME_rec, SUBFME8_rec, SUBFME_rec, ADDZE8_rec, ADDZE_rec,
    SUBFZE_rec, SUBFZE8_rec, ADD8O_rec, SUBFO_rec, SUBF8O_rec, ADD4O_rec,
    ADDEO, ADDE8O, SUBFEO, SUBFE8O, ADDMEO, SUBFMEO, SUBFME8O, ADDME8O,
    ADDZEO, ADDZE8O, SUBFZEO, SUBFZE8O, NEG8O_rec, NEGO_rec,
    ADDE8O_rec, ADDEO_rec, ADDMEO_rec, ADDME8O_rec, SUBFMEO_rec, SUBFME8O_rec,
    ADDZEO_rec, ADDZE8O_rec, SUBFZEO_rec, SUBFZE8O_rec,
    ADDC8_rec, ADDC_rec, ADDCO, ADDCO_rec, ADDC8O, ADDC8O_rec,
    SUBFC8_rec, SUBFC_rec, SUBFCO, SUBFC8O,  SUBFCO_rec, SUBFC8O_rec,
    EXTSB, EXTSB8, EXTSB_rec, EXTSH, EXTSH8, EXTSH_rec, EXTSW, EXTSW_rec,
    RLDICL, RLDICL_rec, RLDICR, RLDICR_rec, RLDIC, RLDIC_rec,
    RLWINM, RLWINM8, RLWINM_rec, RLDCL, RLDCL_rec, RLDCR, RLDCR_rec,
    RLWNM, RLWNM8, RLWNM_rec, RLDIMI, RLDIMI_rec,
    RLDICL_32, RLDICL_32_64, RLDICL_32_rec, RLDICR_32, RLWINM8_rec, RLWNM8_rec,
    SLD, SLD_rec, SLW, SLW8, SLW_rec, SLW8_rec, SRD, SRD_rec, SRW, SRW8, SRW_rec, SRW8_rec,
    SRADI, SRADI_rec, SRAWI, SRAWI_rec, SRAD, SRAD_rec, SRAW, SRAW_rec, SRADI_32,
    SUBFE, SUBFE8, SUBFE8O_rec, SUBFEO_rec
  )>;

  def : InstRW<[P7_VPM_6C, P7_DISP_FP], (instrs
    VPERM,
    XXMRGHW, XXMRGLW, XXPERMDI, XXPERMDIs, XXSLDWI, XXSLDWIs,
    VSPLTB, VSPLTBs, VSPLTH, VSPLTHs, VSPLTISB, VSPLTISH, VSPLTISW, VSPLTW, XXSPLTW, XXSPLTWs,
    VSEL, XXSEL,
    VPKSHSS, VPKSHUS, VPKSWSS, VPKSWUS, VPKUHUM, VPKUHUS, VPKUWUM, VPKUWUS, VPKPX,
    VUPKHPX, VUPKHSB, VUPKHSH, VUPKLPX, VUPKLSB, VUPKLSH
  )>;

  def : InstRW<[P7_VMX_6C, P7_DISP_FP], (instrs
    (instregex "^VADD(U|S)(B|H|W)(S|M)$"),
    (instregex "^V(MAX|MIN)(S|U)(B|H|W)$"),
    (instregex "^V(MRG)(L|H)(B|H|W)$"),
    XXLORf, XXLXORdpz, XXLXORspz, XXLXORz,
    XVRSQRTESP, XVRESP, XVSQRTSP, XVTDIVSP, XVTSQRTSP,
    XVCMPEQSP, XVCMPEQSP_rec, XVCMPGESP, XVCMPGESP_rec, XVCMPGTSP, XVCMPGTSP_rec,
    XVCVSXDSP, XVCVSXWSP, XVCVUXDSP, XVCVUXWSP, XVDIVSP, XVCPSGNSP, XVCVDPSP,
    VADDCUW, VADDFP,
    VAND, VANDC, VAVGSB, VAVGSH, VAVGSW, VAVGUB, VAVGUH, VAVGUW, VCFSX, VCFUX,
    VCMPBFP, VCMPBFP_rec, VCMPEQFP, VCMPEQFP_rec, VCMPEQUB, VCMPEQUB_rec, VCMPEQUH, VCMPEQUH_rec,
    VCMPEQUW, VCMPEQUW_rec, VCMPGEFP, VCMPGEFP_rec, VCMPGTFP, VCMPGTFP_rec, VCMPGTSB, VCMPGTSB_rec,
    VCMPGTSH, VCMPGTSH_rec, VCMPGTSW, VCMPGTSW_rec, VCMPGTUB, VCMPGTUB_rec, VCMPGTUH, VCMPGTUH_rec,
    VCMPGTUW, VCMPGTUW_rec,
    VCTSXS, VCTUXS, VEXPTEFP, VLOGEFP, VNOR, VOR,
    VMADDFP, VMHADDSHS, VMHRADDSHS, VMLADDUHM, VNMSUBFP,
    VMAXFP, VMINFP,
    VMSUMMBM, VMSUMSHM, VMSUMSHS, VMSUMUBM, VMSUMUDM, VMSUMUHM, VMSUMUHS,
    VMULESB, VMULESH, VMULEUB, VMULEUH, VMULOSB, VMULOSH, VMULOUB, VMULOUH,
    VREFP, VRFIM, VRFIN, VRFIP, VRFIZ, VRLB, VRLH, VRLW, VRSQRTEFP,
    VSR, VSRAB, VSRAH, VSRAW,
    VSRB, VSRH, VSRO, VSRW, VSUBCUW, VSL, VSLB, VSLDOI, VSLH, VSLO, VSLW,
    VSUBFP, VSUBSBS, VSUBSHS, VSUBSWS, VSUBUBM, VSUBUBS, VSUBUHM, VSUBUHS, VSUBUWM, VSUBUWS,
    VSUM2SWS, VSUM4SBS, VSUM4SHS, VSUM4UBS, VSUMSWS, VXOR,
    XXLAND, XXLANDC, XXLNOR, XXLOR, XXLXOR
  )>;

  def : InstRW<[P7_FPU_6C, P7_DISP_FP], (instrs
    XVCMPEQDP, XVCMPEQDP_rec, XVCMPGEDP, XVCMPGEDP_rec, XVCMPGTDP, XVCMPGTDP_rec,
    XVCPSGNDP, XVCVDPSXDS, XVCVDPSXWS, XVCVDPUXDS, XVCVDPUXWS, XVCVSPSXDS, XVCVSPSXWS,
    XVCVSPUXDS, XVCVSPUXWS, XVCVSXDDP, XVCVSXWDP, XVCVUXDDP, XVCVUXWDP, XVDIVDP,
    XVRDPI, XVRDPIC, XVRDPIM, XVRDPIP, XVRDPIZ, XVREDP, XVRSPI, XVRSPIC, XVRSPIM, XVRSPIP, XVRSPIZ,
    XVRSQRTEDP, XVSQRTDP, XVTDIVDP, XVTSQRTDP
  )>;

  def : InstRW<[P7_FXU_3C, P7_DISP_FX], (instrs
    CNTLZDM, CNTLZW8_rec, CNTTZD, CNTTZDM, CNTTZD_rec, CNTTZW, CNTTZW8, CNTTZW8_rec, CNTTZW_rec,
    CNTLZD, CNTLZD_rec, CNTLZW, CNTLZW8, CNTLZW_rec
  )>;

  def : InstRW<[P7_FXU_5C, P7_DISP_FX], (instrs
    MULLI, MULLI8, MULLW, MULHW, MULHWU, MULLD, MULHD, MULHDU, MULLWO, MULLDO,
    MULLW_rec, MULLD_rec, MULHD_rec, MULHW_rec, MULHDU_rec, MULHWU_rec, MULLWO_rec, MULLDO_rec
  )>;

  def : InstRW<[P7_FXU_38C, P7_DISP_FX], (instrs
    DIVDE, DIVDEO, DIVDEO_rec, DIVDEU, DIVDEUO, DIVDEUO_rec, DIVDEU_rec, DIVDE_rec,
    DIVWE, DIVWEO, DIVWEO_rec, DIVWEU, DIVWEUO, DIVWEUO_rec, DIVWEU_rec, DIVWE_rec,
    DIVW, DIVWU, DIVWU_rec, DIVWO, DIVWO_rec, DIVWUO, DIVWUO_rec, DIVW_rec
  )>;

  def : InstRW<[P7_FXU_69C, P7_DISP_FX], (instrs
    DIVD, DIVDU, DIVDO, DIVDO_rec, DIVDUO, DIVDUO_rec, DIVDU_rec, DIVD_rec
  )>;

  def : InstRW<[P7_ScalarFPU_6C, P7_DISP_FP], (instrs
    FRE, FRES_rec, FRE_rec, FRSP_rec, FTDIV, FTSQRT,
    FMR, FNEGD, FNEGS, FABSD, FABSS, FNABSD, FNABSS, FADD, FADDS, FSUB, FSUBS, FMUL, FMULS,
    FMADD, FMADDS, FMSUB, FMSUBS, FNMADD, FNMADDS, FNMSUB, FNMSUBS,
    FMADDS_rec, FMADD_rec, FMSUBS_rec, FMSUB_rec, FNMADDS_rec, FNMADD_rec, FNMSUBS_rec, FNMSUB_rec,
    XSNEGDP, XSABSDP, XSNABSDP, XSNABSDPs, XSADDDP, XSADDSP, XSSUBDP, XSSUBSP, XSMULDP, XSMULSP,
    XSMADDADP, XSMADDASP, XSMADDMDP, XSMADDMSP, XSMSUBADP, XSMSUBASP, XSMSUBMDP, XSMSUBMSP,
    XSNMADDADP, XSNMADDASP, XSNMADDMDP, XSNMADDMSP, XSNMSUBADP, XSNMSUBASP, XSNMSUBMDP, XSNMSUBMSP,
    FRSP, FRES, FRSQRTE, FRSQRTES, FRSQRTES_rec, FRSQRTE_rec,
    FCTID, FCTIDU, FCTIDUZ, FCTIDZ, FCTIW, FCTIWU, FCTIWUZ, FCTIWZ, FCFID, FCFIDS, FCFIDU, FCFIDUS,
    FSELD, FSELS, FSELD_rec, FSELS_rec, FABSD_rec, FABSS_rec, FNABSD_rec, FNABSS_rec,
    FMR_rec, FMULS_rec, FMUL_rec, FNEGD_rec, FNEGS_rec, FADDS_rec, FADD_rec, FSUBS_rec, FSUB_rec,
    FCFIDS_rec, FCFIDUS_rec, FCFIDU_rec, FCFID_rec, FCTIDUZ_rec, FCTIDU_rec, FCTIDZ_rec, FCTID_rec,
    FCTIWUZ_rec, FCTIWU_rec, FCTIWZ_rec, FCTIW_rec,
    FCPSGND, FCPSGND_rec, FCPSGNS, FCPSGNS_rec,
    FRIMD, FRIMD_rec, FRIMS, FRIMS_rec, FRIND, FRIND_rec, FRINS, FRINS_rec,
    FRIPD, FRIPD_rec, FRIPS, FRIPS_rec, FRIZD, FRIZD_rec, FRIZS, FRIZS_rec,
    XSCPSGNDP, XSCVDPSP, XSCVDPSXDS, XSCVDPSXDSs, XSCVDPSXWS, XSCVDPSXWSs, XSCVDPUXDS, XSCVDPUXDSs,
    XSCVDPUXWS, XSCVDPUXWSs, XSCVSPDP, XSCVSXDDP, XSCVUXDDP, XSMAXDP, XSMINDP, XSRDPI, XSRDPIC,
    XSRDPIM, XSRDPIP, XSRDPIZ, XSREDP, XSRSQRTEDP,
    XSTDIVDP,
    XSTSQRTDP
  )>;

  def : InstRW<[P7_FPU_6C, P7_DISP_FP], (instrs
    (instregex "^XV(N)?(M)?(ADD|SUB)(A|M)?(D|S)P$"),
    XVMAXDP, XVMAXSP, XVMINDP, XVMINSP, XVMULDP, XVMULSP, XVNABSDP, XVNABSSP, XVNEGDP, XVNEGSP,
    XVABSDP, XVABSSP, XVADDDP, XVADDSP
  )>;

  def : InstRW<[P7_ScalarFPU_8C, P7_DISP_FP], (instrs FCMPOD, FCMPOS, FCMPUD, FCMPUS, XSCMPODP, XSCMPUDP)>;
  def : InstRW<[P7_ScalarFPU_27C, P7_DISP_FP], (instrs FDIVS, XSDIVSP, FDIVS_rec)>;
  def : InstRW<[P7_ScalarFPU_32C, P7_DISP_FP], (instrs FSQRTS, XSSQRTSP, FSQRTS_rec)>;
  def : InstRW<[P7_ScalarFPU_33C, P7_DISP_FP], (instrs FDIV, XSDIVDP, FDIV_rec)>;
  def : InstRW<[P7_ScalarFPU_44C, P7_DISP_FP], (instrs FSQRT, XSSQRTDP, FSQRT_rec)>;
}
