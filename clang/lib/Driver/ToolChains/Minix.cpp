//===--- Minix.cpp - Minix ToolChain Implementations ------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Minix.h"
#include "CommonArgs.h"
#include "clang/Config/config.h"
#include "clang/Driver/Compilation.h"
#include "clang/Driver/Driver.h"
#include "clang/Driver/InputInfo.h"
#include "clang/Driver/Options.h"
#include "llvm/Option/ArgList.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"

using namespace clang::driver;
using namespace clang;
using namespace llvm::opt;

void tools::minix::Assembler::ConstructJob(Compilation &C, const JobAction &JA,
                                           const InputInfo &Output,
                                           const InputInfoList &Inputs,
                                           const ArgList &Args,
                                           const char *LinkingOutput) const {
  claimNoWarnArgs(Args);
  ArgStringList CmdArgs;

  Args.AddAllArgValues(CmdArgs, options::OPT_Wa_COMMA, options::OPT_Xassembler);

  CmdArgs.push_back("-o");
  CmdArgs.push_back(Output.getFilename());

  for (const auto &II : Inputs)
    CmdArgs.push_back(II.getFilename());

  const char *Exec = Args.MakeArgString(getToolChain().GetProgramPath("as"));
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

void tools::minix::Linker::ConstructJob(Compilation &C, const JobAction &JA,
                                 const InputInfo &Output,
                                 const InputInfoList &Inputs,
                                 const ArgList &Args,
                                 const char *LinkingOutput) const {
  const Driver &D = getToolChain().getDriver();
  ArgStringList CmdArgs;

  if (Output.isFilename()) {
    CmdArgs.push_back("-o");
    CmdArgs.push_back(Output.getFilename());
  } else {
    assert(Output.isNothing() && "Invalid output.");
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crt1.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crti.o")));
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtbegin.o")));
    CmdArgs.push_back(Args.MakeArgString(getToolChain().GetFilePath("crtn.o")));
  }

  Args.AddAllArgs(CmdArgs,
                  {options::OPT_L, options::OPT_T_Group, options::OPT_e});

  AddLinkerInputs(getToolChain(), Inputs, Args, CmdArgs, JA);

  getToolChain().addProfileRTLibs(Args, CmdArgs);

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nodefaultlibs,
                   options::OPT_r)) {
    if (D.CCCIsCXX()) {
      if (getToolChain().ShouldLinkCXXStdlib(Args))
        getToolChain().AddCXXStdlibLibArgs(Args, CmdArgs);
      CmdArgs.push_back("-lm");
    }
  }

  if (!Args.hasArg(options::OPT_nostdlib, options::OPT_nostartfiles,
                   options::OPT_r)) {
    if (Args.hasArg(options::OPT_pthread))
      CmdArgs.push_back("-lpthread");
    CmdArgs.push_back("-lc");
    CmdArgs.push_back("-lCompilerRT-Generic");
    CmdArgs.push_back("-L/usr/pkg/compiler-rt/lib");
    CmdArgs.push_back(
        Args.MakeArgString(getToolChain().GetFilePath("crtend.o")));
  }

  const char *Exec = Args.MakeArgString(getToolChain().GetLinkerPath());
  C.addCommand(std::make_unique<Command>(JA, *this,
                                         ResponseFileSupport::AtFileCurCP(),
                                         Exec, CmdArgs, Inputs, Output));
}

/// Minix - Minix tool chain which can call as(1) and ld(1) directly.

toolchains::Minix::Minix(const Driver &D, const llvm::Triple &Triple,
                         const ArgList &Args)
    : Generic_ELF(D, Triple, Args) {
  getFilePaths().push_back(getDriver().Dir + "/../lib");
  getFilePaths().push_back("/usr/lib");
}

void toolchains::Minix::AddClangSystemIncludeArgs(
    const llvm::opt::ArgList &DriverArgs,
    llvm::opt::ArgStringList &CC1Args) const {
  const Driver &D = getDriver();

  if (DriverArgs.hasArg(clang::driver::options::OPT_nostdinc))
    return;

  if (!DriverArgs.hasArg(options::OPT_nobuiltininc)) {
    SmallString<128> Dir(D.ResourceDir);
    llvm::sys::path::append(Dir, "include");
    addSystemInclude(DriverArgs, CC1Args, Dir.str());
  }

  if (DriverArgs.hasArg(options::OPT_nostdlibinc))
    return;

  // Check for configure-time C include directories.
  StringRef CIncludeDirs(C_INCLUDE_DIRS);
  if (CIncludeDirs != "") {
    SmallVector<StringRef, 5> dirs;
    CIncludeDirs.split(dirs, ":");
    for (StringRef dir : dirs) {
      StringRef Prefix =
          llvm::sys::path::is_absolute(dir) ? StringRef(D.SysRoot) : "";
      addExternCSystemInclude(DriverArgs, CC1Args, Prefix + dir);
    }
    return;
  }

  addExternCSystemInclude(DriverArgs, CC1Args,
                          concat(D.SysRoot, "/usr/include"));
}

void toolchains::Minix::addLibStdCxxIncludePaths(const llvm::opt::ArgList &DriverArgs,
                                         llvm::opt::ArgStringList &CC1Args) const {
  addLibStdCXXIncludePaths(concat(getDriver().SysRoot, "/usr/gnu/include/c++/4.4.3"), "", "",
                           DriverArgs, CC1Args);
}

Tool *toolchains::Minix::buildAssembler() const {
  return new tools::minix::Assembler(*this);
}

Tool *toolchains::Minix::buildLinker() const {
  return new tools::minix::Linker(*this);
}
