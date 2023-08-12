//===-- llvm/TargetParser/TripleUtils.h - Triple Utils class--*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_TARGETPARSER_TRIPLE_UTILS_H
#define LLVM_TARGETPARSER_TRIPLE_UTILS_H

#include "llvm/TargetParser/Triple.h"
#include <string>

namespace llvm {

class TripleUtils {
public:
  /// Test whether the architecture is 16-bit
  ///
  /// Note that this tests for 16-bit pointer width, and nothing else.

  static bool isArch16Bit(const llvm::Triple &tripleImpl);
};

} // namespace llvm

#endif