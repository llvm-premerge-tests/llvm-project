//===- llvm/MC/DXContainerPSVInfo.cpp - DXContainer PSVInfo -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/MC/DXContainerPSVInfo.h"
#include "llvm/BinaryFormat/DXContainer.h"
#include "llvm/Support/raw_ostream.h"

using namespace llvm;
using namespace llvm::mcdxbc;
using namespace llvm::dxbc::PSV;

template <typename T>
std::enable_if_t<std::is_integral<T>::value, void>
swappingWrite(raw_ostream &OS, T Val) {
  if (sys::IsBigEndianHost)
    sys::swapByteOrder(Val);
  OS.write(reinterpret_cast<const char *>(&Val), sizeof(T));
}

void PSVRuntimeInfo::write(raw_ostream &OS, uint32_t Version) const {
  uint32_t InfoSize;
  uint32_t BindingSize;
  switch (Version) {
  case 0:
    InfoSize = sizeof(dxbc::PSV::v0::RuntimeInfo);
    BindingSize = sizeof(dxbc::PSV::v0::ResourceBindInfo);
    break;
  case 1:
    InfoSize = sizeof(dxbc::PSV::v1::RuntimeInfo);
    BindingSize = sizeof(dxbc::PSV::v0::ResourceBindInfo);
    break;
  case 2:
  default:
    InfoSize = sizeof(dxbc::PSV::v2::RuntimeInfo);
    BindingSize = sizeof(dxbc::PSV::v2::ResourceBindInfo);
  }
  // Write the size of the info.
  swappingWrite(OS, InfoSize);
  // Write the info itself.
  OS.write(reinterpret_cast<const char *>(&BaseData), InfoSize);

  uint32_t ResourceCount = static_cast<uint32_t>(Resources.size());

  swappingWrite(OS, ResourceCount);
  swappingWrite(OS, BindingSize);

  for (const auto &Res : Resources)
    OS.write(reinterpret_cast<const char *>(&Res), BindingSize);
}
