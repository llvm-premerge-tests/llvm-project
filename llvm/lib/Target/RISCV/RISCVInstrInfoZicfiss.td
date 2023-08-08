//===-------- RISCVInstrInfoZicfiss.td - RISC-V CFG -*- tablegen -*--------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

//===----------------------------------------------------------------------===//
// Operand definitions.
//===----------------------------------------------------------------------===//

def uimm5nonzero : Operand<XLenVT>,
                   ImmLeaf<XLenVT, [{return (Imm != 0) && isUInt<5>(Imm);}]> {
  let ParserMatchClass = UImmAsmOperand<5, "NonZero">;
  let EncoderMethod = "getImmOpValue";
  let DecoderMethod = "decodeUImmNonZeroOperand<5>";
  let OperandType = "OPERAND_UIMM5_NONZERO";
  let OperandNamespace = "RISCVOp";
  let MCOperandPredicate = [{
    int64_t Imm;
    if (MCOp.evaluateAsConstantImm(Imm))
      return (Imm != 0) && isUInt<5>(Imm);
    return MCOp.isBareSymbolRef();
  }];
}

//===----------------------------------------------------------------------===//
// Instruction class templates
//===----------------------------------------------------------------------===//

let Uses = [SSP], hasSideEffects = 0, mayLoad = 1, mayStore = 0 in
class RV_SSPop<bits<5> _rd, bits<5> _rs1, string opcodestr, string argstr> :
  RVInstI<0b100, OPC_SYSTEM, (outs GPR:$rd), (ins GPR:$rs1), opcodestr, argstr> {
  let rd = _rd;
  let rs1 = _rs1;
  let imm12 = 0b100000011100;
}

class RVC_SSInst<bits<5> rs1, string opcodestr, string argstr> :
  RVInst16<(outs), (ins:$rs1), opcodestr, argstr, [], InstFormatOther> {
  let Inst{15-13} = 0b011;
  let Inst{12} = 0;
  let Inst{11-7} = rs1;
  let Inst{6-2} = 0b00000;
  let Inst{1-0} = 0b01;
}

//===----------------------------------------------------------------------===//
// Instructions
//===----------------------------------------------------------------------===//

let Predicates = [HasStdExtZicfiss] in {
def SSLoadX1: RV_SSPop<0b00001, 0b00000, "ssload", "x1">;
def SSLoadX5: RV_SSPop<0b00101, 0b00000, "ssload", "x5">;
let Defs = [SSP] in {
def SSPopChkX1: RV_SSPop<0b00000, 0b00001, "sspopchk", "x1">;
def SSPopChkX5: RV_SSPop<0b00000, 0b00101, "sspopchk", "x5">;
} // Defs = [SSP]

let Uses = [SSP], hasSideEffects = 0, mayLoad = 0, mayStore = 0 in {
def SSPINC : RVInstI<0b100, OPC_SYSTEM, (outs), (ins uimm5nonzero:$rs1),
                     "sspinc", "$rs1"> {
  let imm12 = 0b100000011101;
  let rd = 0b00000;
}

def SSPRR : RVInstR<0b1000011, 0b100, OPC_SYSTEM, (outs GPRNoX0:$rd), (ins),
                    "ssprr", "$rd"> {
  let rs2 = 0b00000;
  let rs1 = 0b00000;
}
} // Uses = [SSP], hasSideEffects = 0, mayLoad = 0, mayStore = 0

let Uses = [SSP], Defs = [SSP], hasSideEffects = 0, mayLoad = 0, mayStore = 1 in {
def SSPUSH : RVInstR<0b1000101, 0b100, OPC_SYSTEM, (outs), (ins GPRRA:$rs2),
                    "sspush", "$rs2"> {
  let rd = 0b00000;
  let rs1 = 0b00000;
}

def SSAMOSWAP :
  RVInstRAtomic<0b10000, 0, 1, 0b100, OPC_SYSTEM, (outs GPRNoX0:$rd),
                (ins GPR:$rs2, GPR:$rs1), "ssamoswap", "$rd, $rs2, (${rs1})">;
} // Uses = [SSP], Defs = [SSP], hasSideEffects = 0, mayLoad = 0, mayStore = 1
} // Predicates = [HasStdExtZicfiss]

let Predicates = [HasStdExtZicfiss, HasStdExtC] in {
let Uses = [SSP], Defs = [SSP], hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def C_SSPUSHX1 : RVC_SSInst<0b00001, "c.sspush", "x1">;

let Uses = [SSP], Defs = [SSP], hasSideEffects = 0, mayLoad = 0, mayStore = 1 in
def C_SSPopChkX5 : RVC_SSInst<0b00101, "c.sspopchk", "x5">;
}// Predicates = [HasStdExtZicfiss, HasStdExtC]
