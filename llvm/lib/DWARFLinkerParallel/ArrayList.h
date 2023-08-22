//===- ArrayList.h ----------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_LIB_DWARFLINKERPARALLEL_ARRAYLIST_H
#define LLVM_LIB_DWARFLINKERPARALLEL_ARRAYLIST_H

#include "DWARFLinkerGlobalData.h"
#include "llvm/Support/PerThreadBumpPtrAllocator.h"

namespace llvm {
namespace dwarflinker_parallel {

/// This class is a simple list of T structures. It keeps elements as
/// pre-allocated groups to save memory for each element's next pointer.
/// It allocates internal data using specified per-thread BumpPtrAllocator.
/// Method add() can be called asynchronously.
template <typename T, size_t ItemsGroupSize = 512> class ArrayList {
public:
  /// Add specified \p Item to the list.
  T &add(const T &Item) {
    assert(Allocator != nullptr);

    // Allocate head group if it is not allocated yet.
    while (LastGroup == nullptr) {
      if (allocateNewGroup(GroupsHead, nullptr))
        LastGroup = GroupsHead.load();
    }

    ItemsGroup *CurGroup;
    size_t CurItemsCount;
    do {
      CurGroup = LastGroup;
      CurItemsCount = CurGroup->ItemsCount.fetch_add(1);

      // Check whether current group is full.
      if (CurItemsCount < ItemsGroupSize)
        break;

      // Allocate next group if necessary.
      if (CurGroup->Next == nullptr)
        allocateNewGroup(CurGroup->Next, nullptr);

      LastGroup.compare_exchange_weak(CurGroup, CurGroup->Next);
    } while (true);

    // Store item into the current group.
    CurGroup->Items[CurItemsCount] = Item;
    return CurGroup->Items[CurItemsCount];
  }

  using ItemHandlerTy = function_ref<void(T &)>;

  /// Enumerate all items and apply specified \p Handler to each.
  void forEach(ItemHandlerTy Handler) {
    for (ItemsGroup *CurGroup = GroupsHead; CurGroup != nullptr;
         CurGroup = CurGroup->Next) {
      for (size_t Idx = 0; Idx < CurGroup->getItemsCount(); Idx++)
        Handler(CurGroup->Items[Idx]);
    }
  }

  /// Check whether list is empty.
  bool empty() { return GroupsHead == nullptr; }

  /// Erase list.
  void erase() {
    GroupsHead = nullptr;
    LastGroup = nullptr;
  }

  void setAllocator(parallel::PerThreadBumpPtrAllocator *Allocator) {
    this->Allocator = Allocator;
  }

protected:
  struct ItemsGroup {
    std::array<T, ItemsGroupSize> Items;
    std::atomic<ItemsGroup *> Next = nullptr;
    std::atomic<size_t> ItemsCount = 0;

    size_t getItemsCount() const {
      return std::min(ItemsCount.load(), ItemsGroupSize);
    }
  };

  bool allocateNewGroup(std::atomic<ItemsGroup *> &AtomicGroup,
                        ItemsGroup *Group) {
    // Allocate new group.
    ItemsGroup *NewGroup = Allocator->Allocate<ItemsGroup>();
    NewGroup->ItemsCount = 0;
    NewGroup->Next = nullptr;

    // Try to replace current group with allocated one.
    if (AtomicGroup.compare_exchange_weak(Group, NewGroup))
      return true;

    // Put allocated group as next group.
    ItemsGroup *NextGroup = Group;
    while (NextGroup != nullptr) {
      NextGroup = NextGroup->Next;

      if (NextGroup == nullptr) {
        if (NextGroup->Next.compare_exchange_weak(NextGroup, NewGroup))
          break;
      }
    }

    return false;
  }

  std::atomic<ItemsGroup *> GroupsHead = nullptr;
  std::atomic<ItemsGroup *> LastGroup = nullptr;
  parallel::PerThreadBumpPtrAllocator *Allocator = nullptr;
};

} // end of namespace dwarflinker_parallel
} // end namespace llvm

#endif // LLVM_LIB_DWARFLINKERPARALLEL_ARRAYLIST_H
