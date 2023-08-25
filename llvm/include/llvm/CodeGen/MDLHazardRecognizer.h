//=- llvm/CodeGen/MDLHazardRecognizer.h - MDL Scheduling Support -*- C++ -*-==//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements the MDLHazardRecognizer class, which implements
// hazard-avoidance heuristics for scheduling using the MDL infrastructure.
//
//===----------------------------------------------------------------------===//

#ifndef LLVM_CODEGEN_MDLHAZARDRECOGNIZER_H
#define LLVM_CODEGEN_MDLHAZARDRECOGNIZER_H

#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/MC/MDLInfo.h"
#include <memory>

namespace llvm {

using namespace mdl;

class MachineInstr;
class SUnit;

/// MDLHazardRecognizer - This uses the MDL to determine if an instruction
/// can be issued in the current cycle.  It uses both the current issue packet,
/// and resources reserved for instructions issued in other cycles.
class MDLHazardRecognizer : public ScheduleHazardRecognizer {
protected:
  const TargetSubtargetInfo *STI;
  CpuInfo *Cpu;
  Reservations *ResSet;
  SlotSet Packet;
  const char *DebugType;

public:
  MDLHazardRecognizer(const TargetSubtargetInfo *STI,
                      const char *ParentDebugType = "")
      : STI(STI), Cpu(STI->getCpuInfo()), ResSet(Cpu->allocReservations()),
        DebugType(ParentDebugType) {
    Packet.reserve(Cpu->getMaxIssue());
  }
  ~MDLHazardRecognizer() { delete ResSet; }

  bool atIssueLimit() const override {
    return Packet.size() == Cpu->getMaxIssue();
  }
  unsigned IssueSize() const override { return Packet.size(); }
  SlotSet *getPacket() override { return &Packet; }

  HazardType getHazardType(SUnit *SU, int Stalls = 0) override {
    if (Stalls == 0)
      return canReserveResources(*SU->getInstr()) ? NoHazard : Hazard;

    auto Res = std::unique_ptr<Reservations>(ResSet->clone());
    SlotDesc Cand(SU->getInstr(), STI);

    for (; Stalls < 0; Stalls++)
      Res->recede();
    for (; Stalls > 0; Stalls--)
      Res->advance();
    return Cpu->canAddToBundle(Packet, Cand, *Res.get()) ? NoHazard : Hazard;
  }

  void Reset() override {
    Packet.clear();
    ResSet->reset();
  }

  void EmitInstruction(SUnit *SU) override {
    SlotDesc Candidate(SU->getInstr(), STI);
    Cpu->addToBundle(Packet, Candidate, *ResSet);
  }

  void EmitInstruction(MachineInstr *MI) override {
    SlotDesc Candidate(MI, STI);
    Cpu->addToBundle(Packet, Candidate, *ResSet);
  }

  void AdvanceCycle() override {
    Cpu->addBundleToReservation(Packet, *ResSet);
    ResSet->advance();
    Packet.clear();
  }

  void RecedeCycle() override {
    Cpu->addBundleToReservation(Packet, *ResSet);
    ResSet->recede();
    Packet.clear();
  }

  // These functions replace Packetizer methods.

  void clearResources() override { ResSet->reset(); }

  bool canReserveResources(MachineInstr &MI) override {
    SlotDesc Candidate(&MI, STI);
    return Cpu->canAddToBundle(Packet, Candidate, *ResSet);
  }
  void reserveResources(MachineInstr &MI) override {
    SlotDesc Candidate(&MI, STI);
    Cpu->addToBundle(Packet, Candidate, *ResSet);
  }
};

/// MDLModuloHazardRecognizer - This uses the MDL to determine if an instruction
/// can be issued in the current cycle.  It uses both the current issue packet,
/// and resources reserved for instructions issued in other cycles.
class MDLModuloHazardRecognizer : public ScheduleHazardRecognizer {
protected:
  const TargetSubtargetInfo *STI;
  int II = 0;
  CpuInfo *Cpu;
  Reservations *ResSet;
  SlotSet *Packets; // Slotsets for every cycle of the loop.
  int Cycle = 0;
  const char *DebugType;

public:
  MDLModuloHazardRecognizer(const TargetSubtargetInfo *STI, int II,
                            const char *ParentDebugType = "")
      : STI(STI), II(II), Cpu(STI->getCpuInfo()),
        ResSet(Cpu->allocModuloReservations(II)), Packets(new SlotSet[II]),
        DebugType(ParentDebugType) {
    for (int ii = 0; ii < II; ii++)
      Packets[ii].reserve(Cpu->getMaxIssue());
  }
  ~MDLModuloHazardRecognizer() {
    delete ResSet;
    delete[] Packets;
  }

  bool atIssueLimit() const override {
    return Packets[Cycle].size() == Cpu->getMaxIssue();
  }

  void setCycle(unsigned II) {
    Cycle = II;
    ResSet->setCycle(Cycle);
  }

  // TODO: This function may be obsolete.
  HazardType getHazardType(SUnit *SU, int Stalls = 0) override {
    if (Stalls == 0)
      return canReserveResources(*SU->getInstr()) ? NoHazard : Hazard;

    SlotDesc Cand(SU->getInstr(), STI);
    setCycle(Cycle + Stalls);
    auto Haz =
        Cpu->canAddToBundle(Packets[Cycle], Cand, *ResSet) ? NoHazard : Hazard;
    setCycle(Cycle);
    return Haz;
  }

  // TODO: This function may be obsolete.
  void Reset() override {
    Packets[Cycle].clear();
    ResSet->reset();
  }

  // TODO: This function may be obsolete.
  void EmitInstruction(SUnit *SU) override {
    SlotDesc Candidate(SU->getInstr(), STI);
    Cpu->addToBundle(Packets[Cycle], Candidate, *ResSet);
  }

  // TODO: This function may be obsolete.
  void EmitInstruction(MachineInstr *MI) override {
    SlotDesc Candidate(MI, STI);
    Cpu->addToBundle(Packets[Cycle], Candidate, *ResSet);
  }

  // These functions replace Packetizer methods.
  void clearResources() override { ResSet->reset(); }

  bool canReserveResources(MachineInstr &MI) override {
    SlotDesc Candidate(&MI, STI);
    return Cpu->canAddToBundle(Packets[Cycle], Candidate, *ResSet);
  }
  void reserveResources(MachineInstr &MI) override {
    SlotDesc Candidate(&MI, STI);
    Cpu->addToBundle(Packets[Cycle], Candidate, *ResSet);
  }
};

} // end namespace llvm

#endif // LLVM_CODEGEN_MDLHAZARDRECOGNIZER_H
