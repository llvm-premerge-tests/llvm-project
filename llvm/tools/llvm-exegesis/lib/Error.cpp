//===-- Error.cpp -----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "Error.h"

namespace llvm {
namespace exegesis {

char ClusteringError::ID;

void ClusteringError::log(raw_ostream &OS) const { OS << Msg; }

std::error_code ClusteringError::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

char SnippetCrash::ID;

void SnippetCrash::log(raw_ostream &OS) const { OS << Msg; }

std::error_code SnippetCrash::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

char SnippetSegfaultCrash::ID;

void SnippetSegfaultCrash::log(raw_ostream &OS) const {
  OS << "A segmentation fault occurred at address " + Twine::utohexstr(Address);
}

std::error_code SnippetSegfaultCrash::convertToErrorCode() const {
  return inconvertibleErrorCode();
}

} // namespace exegesis
} // namespace llvm
