//=----- X86InstrGISel.td - X86 GISel target pseudos -*- tablegen -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// X86 GlobalISel target pseudo instruction definitions. This is kept
// separately from the other tablegen files for organizational purposes, but
// share the same infrastructure.
//
//===----------------------------------------------------------------------===//


class X86GenericInstruction : GenericInstruction {
  let Namespace = "X86";
}

// A pseudo to represent 16 bit funnel left shift modulo 32
def G_FSHL16 : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2, type1:$src3);
  let hasSideEffects = false;
}

// A pseudo to represent 16 bit funnel right shift modulo 32
def G_FSHR16 : X86GenericInstruction {
  let OutOperandList = (outs type0:$dst);
  let InOperandList = (ins type0:$src1, type0:$src2, type1:$src3);
  let hasSideEffects = false;
}

def : GINodeEquiv<G_FSHL16, X86fshl>;
def : GINodeEquiv<G_FSHR16, X86fshr>;
