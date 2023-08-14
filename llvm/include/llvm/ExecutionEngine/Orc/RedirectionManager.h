//===- RedirectionManager.h - Redirection manager interface -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Redirection manager interface that redirects a call to symbol to another.
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/Core.h"

namespace llvm {
namespace orc {

/// Base class for performing redirection of call to symbol to another symbol in
/// runtime.
class RedirectionManager {
public:
  /// Symbol name to symbol definition map.
  using SymbolAddrMap = DenseMap<SymbolStringPtr, ExecutorSymbolDef>;

  virtual ~RedirectionManager() = default;
  /// Change the redirection destination of given symbols to new destination
  /// symbols.
  virtual Error redirect(const SymbolAddrMap &NewDests) = 0;

  /// Change the redirection destination of given symbol to new destination
  /// symbol.
  virtual Error redirect(SymbolStringPtr Symbol, ExecutorSymbolDef NewDest) {
    return redirect({{Symbol, NewDest}});
  }

private:
  virtual void anchor();
};

/// Base class for managing redirectable symbols in which a call
/// gets redirected to another symbol in runtime.
class RedirectableSymbolManager : public RedirectionManager {
public:
  /// Symbol name to symbol definition map.
  using SymbolAddrMap = DenseMap<SymbolStringPtr, ExecutorSymbolDef>;

  /// Create redirectable symbols with given symbol names and initial
  /// desitnation symbols.
  virtual Error createRedirectableSymbols(const SymbolAddrMap &InitialDests,
                                          ResourceTrackerSP RT) = 0;

  /// Create a single redirectable symbol with given symbol name and initial
  /// desitnation symbol.
  virtual Error createRedirectableSymbol(SymbolStringPtr Symbol,
                                         ExecutorSymbolDef InitialDest,
                                         ResourceTrackerSP RT) {
    return createRedirectableSymbols({{Symbol, InitialDest}}, RT);
  }
};

} // namespace orc
} // namespace llvm
