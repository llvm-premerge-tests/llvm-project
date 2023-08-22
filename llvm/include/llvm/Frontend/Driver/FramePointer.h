//===-- Clang.cpp - Clang+LLVM ToolChain Implementations --------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "../../../../../clang/include/clang/Basic/CodeGenOptions.h"
#include "../../../../../clang/include/clang/Driver/Options.h"
#include "../../../../../clang/lib/Driver/ToolChains/CommonArgs.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/CodeGen.h"
#include "llvm/TargetParser/Triple.h"

using namespace llvm::opt;

static bool useFramePointerForTargetByDefault(const ArgList &Args,
                                              const llvm::Triple &Triple) {
  if (Args.hasArg(clang::driver::options::OPT_pg) &&
      !Args.hasArg(clang::driver::options::OPT_mfentry))
    return true;

  if (Triple.isAndroid()) {
    switch (Triple.getArch()) {
    case llvm::Triple::aarch64:
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
    case llvm::Triple::riscv64:
      return true;
    default:
      break;
    }
  }

  switch (Triple.getArch()) {
  case llvm::Triple::xcore:
  case llvm::Triple::wasm32:
  case llvm::Triple::wasm64:
  case llvm::Triple::msp430:
    // XCore never wants frame pointers, regardless of OS.
    // WebAssembly never wants frame pointers.
    return false;
  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::riscv32:
  case llvm::Triple::riscv64:
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::sparcv9:
  case llvm::Triple::amdgcn:
  case llvm::Triple::r600:
  case llvm::Triple::csky:
  case llvm::Triple::loongarch32:
  case llvm::Triple::loongarch64:
    return !clang::driver::tools::areOptimizationsEnabled(Args);
  default:
    break;
  }

  if (Triple.isOSFuchsia() || Triple.isOSNetBSD()) {
    return !clang::driver::tools::areOptimizationsEnabled(Args);
  }

  if (Triple.isOSLinux() || Triple.getOS() == llvm::Triple::CloudABI ||
      Triple.isOSHurd()) {
    switch (Triple.getArch()) {
    // Don't use a frame pointer on linux if optimizing for certain targets.
    case llvm::Triple::arm:
    case llvm::Triple::armeb:
    case llvm::Triple::thumb:
    case llvm::Triple::thumbeb:
    case llvm::Triple::mips64:
    case llvm::Triple::mips64el:
    case llvm::Triple::mips:
    case llvm::Triple::mipsel:
    case llvm::Triple::systemz:
    case llvm::Triple::x86:
    case llvm::Triple::x86_64:
      return !clang::driver::tools::areOptimizationsEnabled(Args);
    default:
      return true;
    }
  }

  if (Triple.isOSWindows()) {
    switch (Triple.getArch()) {
    case llvm::Triple::x86:
      return !clang::driver::tools::areOptimizationsEnabled(Args);
    case llvm::Triple::x86_64:
      return Triple.isOSBinFormatMachO();
    case llvm::Triple::arm:
    case llvm::Triple::thumb:
      // Windows on ARM builds with FPO disabled to aid fast stack walking
      return true;
    default:
      // All other supported Windows ISAs use xdata unwind information, so frame
      // pointers are not generally useful.
      return false;
    }
  }

  return true;
}

static bool mustUseNonLeafFramePointerForTarget(const llvm::Triple &Triple) {
  switch (Triple.getArch()) {
  default:
    return false;
  case llvm::Triple::arm:
  case llvm::Triple::thumb:
    // ARM Darwin targets require a frame pointer to be always present to aid
    // offline debugging via backtraces.
    return Triple.isOSDarwin();
  }
}

static clang::CodeGenOptions::FramePointerKind
getFramePointerKind(const ArgList &Args, const llvm::Triple &Triple) {
  // We have 4 states:
  //
  //  00) leaf retained, non-leaf retained
  //  01) leaf retained, non-leaf omitted (this is invalid)
  //  10) leaf omitted, non-leaf retained
  //      (what -momit-leaf-frame-pointer was designed for)
  //  11) leaf omitted, non-leaf omitted
  //
  //  "omit" options taking precedence over "no-omit" options is the only way
  //  to make 3 valid states representable
  Arg *A = Args.getLastArg(clang::driver::options::OPT_fomit_frame_pointer,
                           clang::driver::options::OPT_fno_omit_frame_pointer);

  bool OmitFP = A && A->getOption().matches(
                         clang::driver::options::OPT_fomit_frame_pointer);
  bool NoOmitFP = A && A->getOption().matches(
                           clang::driver::options::OPT_fno_omit_frame_pointer);
  bool OmitLeafFP =
      Args.hasFlag(clang::driver::options::OPT_momit_leaf_frame_pointer,
                   clang::driver::options::OPT_mno_omit_leaf_frame_pointer,
                   Triple.isAArch64() || Triple.isPS() || Triple.isVE() ||
                       (Triple.isAndroid() && Triple.isRISCV64()));
  if (NoOmitFP || mustUseNonLeafFramePointerForTarget(Triple) ||
      (!OmitFP && useFramePointerForTargetByDefault(Args, Triple))) {
    if (OmitLeafFP)
      return clang::CodeGenOptions::FramePointerKind::NonLeaf;
    return clang::CodeGenOptions::FramePointerKind::All;
  }
  return clang::CodeGenOptions::FramePointerKind::None;
}
