//===- DWARFLinkerRelocs.h ---------------------------------------*- C++
//-*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_DWARFLINKER_DWARFLINKERRELOCS_H
#define LLVM_DWARFLINKER_DWARFLINKERRELOCS_H

#include "llvm/Support/Casting.h"

namespace llvm {

/// ValidRelocation represents one relocation entry described by the RelocMap.
class ValidRelocation {
public:
  enum ValidRelocationKind { VRK_ValidReloc };

private:
  ValidRelocationKind Kind;

public:
  ValidRelocationKind getKind() const { return Kind; }

  ValidRelocation(ValidRelocationKind K) : Kind(K) {}
};

/// RelocMap represents information about the relocations of the object
/// files linked together to produce an executable.
class RelocMap {
public:
  virtual ~RelocMap() = default;

  /// This function adds a ValidRelocation to this relocation map.
  virtual void addRelocationMapEntry(const ValidRelocation *Relocation) = 0;
};

} // end namespace llvm

#endif // LLVM_DWARFLINKER_DWARFLINKERRELOCS_H
