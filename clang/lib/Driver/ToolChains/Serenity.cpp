//===---- Serenity.cpp - SerenityOS ToolChain Implementation ----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Serenity.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/Options.h"
#include "clang/Driver/SanitizerArgs.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <string>

using namespace clang::driver;
using namespace clang::driver::toolchains;
using namespace clang;
using namespace llvm::opt;

static bool getPIE(const ArgList &Args, const ToolChain &TC) {
  if (Args.hasArg(options::OPT_static, options::OPT_shared,
                  options::OPT_static_pie))
    return false;
  Arg *Last = Args.getLastArg(options::OPT_pie, options::OPT_no_pie);
  return Last ? Last->getOption().matches(options::OPT_pie) : true;
}

void tools::serenity::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const ArgList &Args,
                                           const char *LinkingOutput) const {
  const auto &TC = getToolChain();
  const auto &D = TC.getDriver();
  const bool IsShared = Args.hasArg(options::OPT_shared);
  const bool IsStatic =
      Args.hasArg(options::OPT_static) && !Args.hasArg(options::OPT_static_pie);
  const bool IsRdynamic = Args.hasArg(options::OPT_rdynamic);
  const bool IsStaticPIE = Args.hasArg(options::OPT_static_pie);
  const bool IsPIE = getPIE(Args, TC);
  ArgStringList CmdArgs;

  if (!D.SysRoot.empty())
    CmdArgs.push_back(Args.MakeArgString("--sysroot=" + D.SysRoot));

  if (IsPIE || IsStaticPIE)
    CmdArgs.push_back("-pie");

  if (IsShared)
    CmdArgs.push_back("-shared");

  if (IsStatic || IsStaticPIE)
    CmdArgs.push_back("-static");

  if (IsStaticPIE) {
    CmdArgs.push_back("--no-dynamic-linker");
    CmdArgs.push_back("-z");
    CmdArgs.push_back("text");
  }

  if (!IsStatic && !IsStaticPIE) {
    if (IsRdynamic)
      CmdArgs.push_back("-export-dynamic");
    CmdArgs.push_back("-dynamic-linker");
    CmdArgs.push_back("/usr/lib/Loader.so");
  }

  CmdArgs.push_back("--eh-frame-hdr");

  assert((Output.isFilename() || Output.isNothing()) && "Invalid output.");
  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  }

  CmdArgs.push_back("-z");
  CmdArgs.push_back("pack-relative-relocs");

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    CmdArgs.push_back(Args.MakeArgString(
        TC.GetFilePath((IsShared) ? "crt0_shared.o" : "crt0.o")));
    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crti.o")));

    std::string crtbegin_path;
    if (TC.GetRuntimeLibType(Args) == ToolChain::RLT_CompilerRT) {
      std::string crtbegin =
          TC.getCompilerRT(Args, "crtbegin", ToolChain::FT_Object);
      if (TC.getVFS().exists(crtbegin))
        crtbegin_path = crtbegin;
    }
    if (crtbegin_path.empty()) {
      const char *crtbegin = (IsShared || IsPIE) ? "crtbeginS.o" : "crtbegin.o";
      crtbegin_path = TC.GetFilePath(crtbegin);
    }
    CmdArgs.push_back(Args.MakeArgString(crtbegin_path));
  }

  Args.addAllArgs(CmdArgs, {options::OPT_L, options::OPT_u});

  TC.AddFilePathLibArgs(Args, CmdArgs);

  if (D.isUsingLTO()) {
    assert(!Inputs.empty() && "Must have at least one input.");
    // Find the first filename InputInfo object.
    auto const* Input = llvm::find_if(
        Inputs, [](const InputInfo &II) -> bool { return II.isFilename(); });
    if (Input == Inputs.end())
      // For a very rare case, all of the inputs to the linker are
      // InputArg. If that happens, just use the first InputInfo.
      Input = Inputs.begin();

    addLTOOptions(TC, Args, CmdArgs, Output, *Input,
                  D.getLTOMode() == LTOK_Thin);
  }

  Args.addAllArgs(CmdArgs, {options::OPT_T_Group, options::OPT_s,
                            options::OPT_t, options::OPT_r});

  addLinkerCompressDebugSectionsOption(TC, Args, CmdArgs);

  AddLinkerInputs(TC, Inputs, Args, CmdArgs, JA);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs,
                   options::OPT_r)) {
    AddRunTimeLibs(TC, D, CmdArgs, Args);

    // We supply our own sanitizer runtimes that output errors to the
    // Kernel debug log as well as stderr.
    // FIXME: Properly port clang/gcc sanitizers and use those instead.
    const SanitizerArgs &Sanitize = TC.getSanitizerArgs(Args);
    if (Sanitize.needsUbsanRt())
      CmdArgs.push_back("-lubsan");
  }

  if (D.CCCIsCXX() && TC.ShouldLinkCXXStdlib(Args)) {
    bool OnlyLibstdcxxStatic = Args.hasArg(options::OPT_static_libstdcxx) &&
                               !Args.hasArg(options::OPT_static);
    CmdArgs.push_back("--push-state");
    CmdArgs.push_back("--as-needed");
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bstatic");
    TC.AddCXXStdlibLibArgs(Args, CmdArgs);
    if (OnlyLibstdcxxStatic)
      CmdArgs.push_back("-Bdynamic");
    CmdArgs.push_back("--pop-state");
  }

  // Silence warnings when linking C code with a C++ '-stdlib' argument.
  Args.ClaimAllArgs(options::OPT_stdlib_EQ);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs,
                   options::OPT_r)) {
    if (!Args.hasArg(options::OPT_nolibc))
      CmdArgs.push_back("-lc");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    std::string crtend_path;
    if (TC.GetRuntimeLibType(Args) == ToolChain::RLT_CompilerRT) {
      std::string crtend =
          TC.getCompilerRT(Args, "crtend", ToolChain::FT_Object);
      if (TC.getVFS().exists(crtend))
        crtend_path = crtend;
    }
    if (crtend_path.empty()) {
      const char *crtend = (IsShared || IsPIE) ? "crtendS.o" : "crtend.o";
      crtend_path = TC.GetFilePath(crtend);
    }
    CmdArgs.push_back(Args.MakeArgString(crtend_path));

    CmdArgs.push_back(Args.MakeArgString(TC.GetFilePath("crtn.o")));
  }

  const char *Exec = Args.MakeArgString(TC.GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

Serenity::Serenity(const Driver &D, const llvm::Triple &Triple,
                   const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(concat(getDriver().SysRoot, "/usr/lib"));
}

Tool *Serenity::buildLinker() const {
  return new tools::serenity::Linker(*this);
}

void Serenity::AddClangSystemIncludeArgs(const ArgList &DriverArgs,
                                         ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc))
    addSystemInclude(DriverArgs, CC1Args, concat(D.ResourceDir, "/include"));

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  addSystemInclude(DriverArgs, CC1Args, concat(D.SysRoot, "/usr/include"));
}
