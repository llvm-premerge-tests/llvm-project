//===-- JITLinkRedirectableSymbolManager.cpp - JITLink redirection in Orc -===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/ExecutionEngine/Orc/JITLinkRedirectableSymbolManager.h"
#include "llvm/ExecutionEngine/Orc/Core.h"

#define DEBUG_TYPE "orc"

using namespace llvm;
using namespace llvm::orc;

Error JITLinkRedirectableSymbolManager::createRedirectableSymbols(
    ResourceTrackerSP RT, const SymbolAddrMap &InitialDests) {
  std::unique_lock<std::mutex> Lock(Mutex);
  if (GetNumAvailableStubs() < InitialDests.size())
    if (auto Err = grow(InitialDests.size() - GetNumAvailableStubs()))
      return Err;

  JITDylib &TargetJD = RT->getJITDylib();
  SymbolMap NewSymbolDefs;
  std::vector<SymbolStringPtr> Symbols;
  for (auto &[K, V] : InitialDests) {
    StubHandle StubID = AvailableStubs.back();
    if (SymbolToStubs[&TargetJD].count(K))
      return make_error<StringError>(
          "Tried to create duplicate redirectable symbols",
          inconvertibleErrorCode());
    SymbolToStubs[&TargetJD][K] = StubID;
    NewSymbolDefs[K] = JumpStubs[StubID];
    Symbols.push_back(K);
    AvailableStubs.pop_back();
  }

  if (auto Err = TargetJD.define(absoluteSymbols(NewSymbolDefs), RT))
    return Err;

  if (auto Err = redirectInner(TargetJD, InitialDests))
    return Err;

  auto Err = RT->withResourceKeyDo([&](ResourceKey Key) {
    TrackedResources[Key].insert(TrackedResources[Key].end(), Symbols.begin(),
                                 Symbols.end());
  });
  if (Err)
    return Err;

  return Error::success();
}

Error JITLinkRedirectableSymbolManager::redirect(
    JITDylib &TargetJD, const SymbolAddrMap &NewDests) {
  std::unique_lock<std::mutex> Lock(Mutex);
  return redirectInner(TargetJD, NewDests);
}

Error JITLinkRedirectableSymbolManager::redirectInner(
    JITDylib &TargetJD, const SymbolAddrMap &NewDests) {
  std::vector<std::pair<ExecutorAddr, ExecutorAddr>> PtrWrites;
  for (auto &[K, V] : NewDests) {
    if (!SymbolToStubs[&TargetJD].count(K))
      return make_error<StringError>(
          "Tried to redirect non-existent redirectalbe symbol",
          inconvertibleErrorCode());
    StubHandle StubID = SymbolToStubs[&TargetJD].at(K);
    PtrWrites.push_back({StubPointers[StubID].getAddress(), V.getAddress()});
  }

  if (ES.getTargetTriple().isArch64Bit()) {
    std::vector<tpctypes::UInt64Write> NativeWrites;
    for (auto &[Ptr, Target] : PtrWrites)
      NativeWrites.push_back(tpctypes::UInt64Write(Ptr, Target.getValue()));
    if (auto Err =
            ES.getExecutorProcessControl().getMemoryAccess().writeUInt64s(
                NativeWrites))
      return Err;
  } else {
    assert(DL.getPointerSize() == 4 && "Unsupported pointer size");
    std::vector<tpctypes::UInt32Write> NativeWrites;
    for (auto &[Ptr, Target] : PtrWrites)
      NativeWrites.push_back(tpctypes::UInt32Write(Ptr, Target.getValue()));
    if (auto Err =
            ES.getExecutorProcessControl().getMemoryAccess().writeUInt32s(
                NativeWrites))
      return Err;
  }
  return Error::success();
}

Error JITLinkRedirectableSymbolManager::grow(unsigned Need) {
  unsigned OldSize = JumpStubs.size();
  unsigned NumNewStubs = alignTo(Need, StubBlockSize);
  unsigned NewSize = OldSize + NumNewStubs;

  JumpStubs.resize(NewSize);
  StubPointers.resize(NewSize);
  AvailableStubs.reserve(NewSize);

  SymbolLookupSet LookupSymbols;
  DenseMap<SymbolStringPtr, ExecutorSymbolDef *> NewDefsMap;

  Triple TT = ES.getTargetTriple();
  auto G = std::make_unique<jitlink::LinkGraph>(
      "<INDIRECT STUBS>", TT, TT.isArch64Bit() ? 8 : 4,
      TT.isLittleEndian() ? support::little : support::big,
      jitlink::getGenericEdgeKindName);
  auto &PointerSection =
      G->createSection(StubPtrTableName, MemProt::Write | MemProt::Read);
  auto &StubsSection =
      G->createSection(JumpStubTableName, MemProt::Exec | MemProt::Read);

  for (size_t I = OldSize; I < NewSize; I++) {
    auto Pointer = AnonymousPtrCreator(*G, PointerSection, nullptr, 0);
    if (auto Err = Pointer.takeError())
      return Err;

    StringRef PtrSymName = StubPtrSymbolName(I);
    Pointer->setName(PtrSymName);
    Pointer->setScope(jitlink::Scope::Default);
    LookupSymbols.add(ES.intern(PtrSymName));
    NewDefsMap[ES.intern(PtrSymName)] = &StubPointers[I];

    auto Stub = PtrJumpStubCreator(*G, StubsSection, *Pointer);
    if (auto Err = Stub.takeError())
      return Err;

    StringRef JumpStubSymName = JumpStubSymbolName(I);
    Stub->setName(JumpStubSymName);
    Stub->setScope(jitlink::Scope::Default);
    LookupSymbols.add(ES.intern(JumpStubSymName));
    NewDefsMap[ES.intern(JumpStubSymName)] = &JumpStubs[I];
  }

  if (auto Err = ObjLinkingLayer.add(JD, std::move(G)))
    return Err;

  auto LookupResult = ES.lookup(makeJITDylibSearchOrder(&JD), LookupSymbols);
  if (auto Err = LookupResult.takeError())
    return Err;

  for (auto &[K, V] : *LookupResult)
    *NewDefsMap.at(K) = V;

  for (size_t I = OldSize; I < NewSize; I++)
    AvailableStubs.push_back(I);

  return Error::success();
}

Error JITLinkRedirectableSymbolManager::handleRemoveResources(
    JITDylib &TargetJD, ResourceKey K) {
  std::unique_lock<std::mutex> Lock(Mutex);
  for (auto &Symbol : TrackedResources[K]) {
    if (!SymbolToStubs[&TargetJD].count(Symbol))
      return make_error<StringError>(
          "Tried to remove non-existent redirectable symbol",
          inconvertibleErrorCode());
    AvailableStubs.push_back(SymbolToStubs[&TargetJD].at(Symbol));
    SymbolToStubs[&TargetJD].erase(Symbol);
    if (SymbolToStubs[&TargetJD].empty())
      SymbolToStubs.erase(&TargetJD);
  }
  TrackedResources.erase(K);

  return Error::success();
}

void JITLinkRedirectableSymbolManager::handleTransferResources(
    JITDylib &TargetJD, ResourceKey DstK, ResourceKey SrcK) {
  std::unique_lock<std::mutex> Lock(Mutex);
  TrackedResources[DstK].insert(TrackedResources[DstK].end(),
                                TrackedResources[SrcK].begin(),
                                TrackedResources[SrcK].end());
  TrackedResources.erase(SrcK);
}
