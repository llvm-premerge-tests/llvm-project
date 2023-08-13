//===--- TripleUtils.cpp - Triple Utils class --------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/TargetParser/TripleUtils.h"
#include "llvm/Support/ErrorHandling.h"
using namespace llvm;

static unsigned getArchPointerBitWidth(llvm::Triple::ArchType Arch) {
  switch (Arch) {
  case llvm::Triple::UnknownArch:
    return 0;

  case llvm::Triple::avr:
  case llvm::Triple::msp430:
    return 16;

  case llvm::Triple::aarch64_32:
  case llvm::Triple::amdil:
  case llvm::Triple::arc:
  case llvm::Triple::arm:
  case llvm::Triple::armeb:
  case llvm::Triple::csky:
  case llvm::Triple::dxil:
  case llvm::Triple::hexagon:
  case llvm::Triple::hsail:
  case llvm::Triple::kalimba:
  case llvm::Triple::lanai:
  case llvm::Triple::le32:
  case llvm::Triple::loongarch32:
  case llvm::Triple::m68k:
  case llvm::Triple::mips:
  case llvm::Triple::mipsel:
  case llvm::Triple::nvptx:
  case llvm::Triple::ppc:
  case llvm::Triple::ppcle:
  case llvm::Triple::r600:
  case llvm::Triple::renderscript32:
  case llvm::Triple::riscv32:
  case llvm::Triple::shave:
  case llvm::Triple::sparc:
  case llvm::Triple::sparcel:
  case llvm::Triple::spir:
  case llvm::Triple::spirv32:
  case llvm::Triple::tce:
  case llvm::Triple::tcele:
  case llvm::Triple::thumb:
  case llvm::Triple::thumbeb:
  case llvm::Triple::wasm32:
  case llvm::Triple::x86:
  case llvm::Triple::xcore:
  case llvm::Triple::xtensa:
    return 32;

  case llvm::Triple::aarch64:
  case llvm::Triple::aarch64_be:
  case llvm::Triple::amdgcn:
  case llvm::Triple::amdil64:
  case llvm::Triple::bpfeb:
  case llvm::Triple::bpfel:
  case llvm::Triple::hsail64:
  case llvm::Triple::le64:
  case llvm::Triple::loongarch64:
  case llvm::Triple::mips64:
  case llvm::Triple::mips64el:
  case llvm::Triple::nvptx64:
  case llvm::Triple::ppc64:
  case llvm::Triple::ppc64le:
  case llvm::Triple::renderscript64:
  case llvm::Triple::riscv64:
  case llvm::Triple::sparcv9:
  case llvm::Triple::spir64:
  case llvm::Triple::spirv64:
  case llvm::Triple::systemz:
  case llvm::Triple::ve:
  case llvm::Triple::wasm64:
  case llvm::Triple::x86_64:
    return 64;
  }
  llvm_unreachable("Invalid architecture value");
}

bool TripleUtils::isArch16Bit(const llvm::Triple &tripleImpl) {
  return getArchPointerBitWidth(tripleImpl.getArch()) == 16;
}

bool TripleUtils::isArch32Bit(const llvm::Triple &tripleImpl) {
  return getArchPointerBitWidth(tripleImpl.getArch()) == 32;
}