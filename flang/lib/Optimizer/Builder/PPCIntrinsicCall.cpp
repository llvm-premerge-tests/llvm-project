//===-- PPCIntrinsicCall.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Helper routines for constructing the FIR dialect of MLIR for PowerPC
// intrinsics. Extensive use of MLIR interfaces and MLIR's coding style
// (https://mlir.llvm.org/getting_started/DeveloperGuide/) is used in this
// module.
//
//===----------------------------------------------------------------------===//

#include "flang/Optimizer/Builder/PPCIntrinsicCall.h"
#include "flang/Evaluate/common.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/MutableBox.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"

namespace fir {

using PI = PPCIntrinsicLibrary;

// PPC specific intrinsic handlers.
static constexpr IntrinsicHandler ppcHandlers[]{
    {"__ppc_mma_assemble_acc",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::AssembleAcc, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"arg1", asValue},
       {"arg2", asValue},
       {"arg3", asValue},
       {"arg4", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_assemble_pair",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::AssemblePair, MMAHandlerOp::SubToFunc>),
     {{{"pair", asAddr}, {"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_build_acc",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::AssembleAcc,
                         MMAHandlerOp::SubToFuncReverseArgOnLE>),
     {{{"acc", asAddr},
       {"arg1", asValue},
       {"arg2", asValue},
       {"arg3", asValue},
       {"arg4", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_disassemble_acc",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::DisassembleAcc, MMAHandlerOp::SubToFunc>),
     {{{"data", asAddr}, {"acc", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_disassemble_pair",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::DisassemblePair, MMAHandlerOp::SubToFunc>),
     {{{"data", asAddr}, {"pair", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvbf16ger2_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvbf16ger2, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvbf16ger2nn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvbf16ger2nn,
                         MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvbf16ger2np",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvbf16ger2np,
                         MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvbf16ger2pn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvbf16ger2pn,
                         MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvbf16ger2pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvbf16ger2pp,
                         MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf16ger2_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf16ger2, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf16ger2nn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf16ger2nn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf16ger2np",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf16ger2np, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf16ger2pn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf16ger2pn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf16ger2pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf16ger2pp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf32ger",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf32ger, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf32gernn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf32gernn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf32gernp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf32gernp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf32gerpn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf32gerpn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf32gerpp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf32gerpp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf64ger",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf64ger, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf64gernn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf64gernn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf64gernp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf64gernp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf64gerpn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf64gerpn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvf64gerpp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvf64gerpp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvi16ger2_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvi16ger2, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvi16ger2pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvi16ger2pp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvi16ger2s",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvi16ger2s, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvi16ger2spp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvi16ger2spp,
                         MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvi4ger8_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvi4ger8, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvi4ger8pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvi4ger8pp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvi8ger4_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvi8ger4, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvi8ger4pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvi8ger4pp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_pmxvi8ger4spp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Pmxvi8ger4spp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr},
       {"a", asValue},
       {"b", asValue},
       {"xmask", asValue},
       {"ymask", asValue},
       {"pmask", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvbf16ger2_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvbf16ger2, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvbf16ger2nn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvbf16ger2nn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvbf16ger2np",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvbf16ger2np, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvbf16ger2pn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvbf16ger2pn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvbf16ger2pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvbf16ger2pp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf16ger2_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf16ger2, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf16ger2nn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf16ger2nn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf16ger2np",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf16ger2np, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf16ger2pn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf16ger2pn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf16ger2pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf16ger2pp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf32ger",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf32ger, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf32gernn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf32gernn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf32gernp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf32gernp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf32gerpn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf32gerpn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf32gerpp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf32gerpp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf64ger",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf64ger, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf64gernn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf64gernn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf64gernp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf64gernp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf64gerpn",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf64gerpn, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvf64gerpp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvf64gerpp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvi16ger2_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvi16ger2, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvi16ger2pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvi16ger2pp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvi16ger2s",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvi16ger2s, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvi16ger2spp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvi16ger2spp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvi4ger8_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvi4ger8, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvi4ger8pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvi4ger8pp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvi8ger4_",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvi8ger4, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvi8ger4pp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvi8ger4pp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xvi8ger4spp",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xvi8ger4spp, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}, {"a", asValue}, {"b", asValue}}},
     /*isElemental=*/true},
    {"__ppc_mma_xxmfacc",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xxmfacc, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}}},
     /*isElemental=*/true},
    {"__ppc_mma_xxmtacc",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xxmtacc, MMAHandlerOp::FirstArgIsResult>),
     {{{"acc", asAddr}}},
     /*isElemental=*/true},
    {"__ppc_mma_xxsetaccz",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(
         &PI::genMmaIntr<MMAOp::Xxsetaccz, MMAHandlerOp::SubToFunc>),
     {{{"acc", asAddr}}},
     /*isElemental=*/true},
    {"__ppc_mtfsf",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(&PI::genMtfsf<false>),
     {{{"mask", asValue}, {"r", asValue}}},
     /*isElemental=*/false},
    {"__ppc_mtfsfi",
     static_cast<IntrinsicLibrary::SubroutineGenerator>(&PI::genMtfsf<true>),
     {{{"bf", asValue}, {"i", asValue}}},
     /*isElemental=*/false},
    {"__ppc_vec_abs",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(&PI::genVecAbs),
     {{{"arg1", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_add",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::Add>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_and",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::And>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_any_ge",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAnyCompare<VecOp::Anyge>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cmpge",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecCmp<VecOp::Cmpge>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cmpgt",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecCmp<VecOp::Cmpgt>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cmple",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecCmp<VecOp::Cmple>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cmplt",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecCmp<VecOp::Cmplt>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_convert",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecConvert<VecOp::Convert>),
     {{{"v", asValue}, {"mold", asValue}}},
     /*isElemental=*/false},
    {"__ppc_vec_ctf",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecConvert<VecOp::Ctf>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_cvf",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecConvert<VecOp::Cvf>),
     {{{"arg1", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_msub",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecNmaddMsub<VecOp::Msub>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_mul",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::Mul>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_nmadd",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecNmaddMsub<VecOp::Nmadd>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sel",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(&PI::genVecSel),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sl",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sl>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sld",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sld>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sldw",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sldw>),
     {{{"arg1", asValue}, {"arg2", asValue}, {"arg3", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sll",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sll>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_slo",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Slo>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sr",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sr>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_srl",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Srl>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sro",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecShift<VecOp::Sro>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_sub",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::Sub>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
    {"__ppc_vec_xor",
     static_cast<IntrinsicLibrary::ExtendedGenerator>(
         &PI::genVecAddAndMulSubXor<VecOp::Xor>),
     {{{"arg1", asValue}, {"arg2", asValue}}},
     /*isElemental=*/true},
};

static constexpr MathOperation ppcMathOperations[] = {
    // fcfi is just another name for fcfid, there is no llvm.ppc.fcfi.
    {"__ppc_fcfi", "llvm.ppc.fcfid", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fcfid", "llvm.ppc.fcfid", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fcfud", "llvm.ppc.fcfud", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctid", "llvm.ppc.fctid", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctidz", "llvm.ppc.fctidz", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctiw", "llvm.ppc.fctiw", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctiwz", "llvm.ppc.fctiwz", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctudz", "llvm.ppc.fctudz", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fctuwz", "llvm.ppc.fctuwz", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fmadd", "llvm.fma.f32",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genMathOp<mlir::math::FmaOp>},
    {"__ppc_fmadd", "llvm.fma.f64",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genMathOp<mlir::math::FmaOp>},
    {"__ppc_fmsub", "llvm.ppc.fmsubs",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_fmsub", "llvm.ppc.fmsub",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fnabs", "llvm.ppc.fnabss", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_fnabs", "llvm.ppc.fnabs", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fnmadd", "llvm.ppc.fnmadds",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_fnmadd", "llvm.ppc.fnmadd",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fnmsub", "llvm.ppc.fnmsub.f32",
     genFuncType<Ty::Real<4>, Ty::Real<4>, Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_fnmsub", "llvm.ppc.fnmsub.f64",
     genFuncType<Ty::Real<8>, Ty::Real<8>, Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fre", "llvm.ppc.fre", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_fres", "llvm.ppc.fres", genFuncType<Ty::Real<4>, Ty::Real<4>>,
     genLibCall},
    {"__ppc_frsqrte", "llvm.ppc.frsqrte", genFuncType<Ty::Real<8>, Ty::Real<8>>,
     genLibCall},
    {"__ppc_frsqrtes", "llvm.ppc.frsqrtes",
     genFuncType<Ty::Real<4>, Ty::Real<4>>, genLibCall},
    {"__ppc_vec_madd", "llvm.fma.v4f32",
     genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>,
                 Ty::RealVector<4>>,
     genLibCall},
    {"__ppc_vec_madd", "llvm.fma.v2f64",
     genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>,
                 Ty::RealVector<8>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxsb",
     genFuncType<Ty::IntegerVector<1>, Ty::IntegerVector<1>,
                 Ty::IntegerVector<1>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxsh",
     genFuncType<Ty::IntegerVector<2>, Ty::IntegerVector<2>,
                 Ty::IntegerVector<2>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxsw",
     genFuncType<Ty::IntegerVector<4>, Ty::IntegerVector<4>,
                 Ty::IntegerVector<4>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxsd",
     genFuncType<Ty::IntegerVector<8>, Ty::IntegerVector<8>,
                 Ty::IntegerVector<8>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxub",
     genFuncType<Ty::UnsignedVector<1>, Ty::UnsignedVector<1>,
                 Ty::UnsignedVector<1>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxuh",
     genFuncType<Ty::UnsignedVector<2>, Ty::UnsignedVector<2>,
                 Ty::UnsignedVector<2>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxuw",
     genFuncType<Ty::UnsignedVector<4>, Ty::UnsignedVector<4>,
                 Ty::UnsignedVector<4>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.altivec.vmaxud",
     genFuncType<Ty::UnsignedVector<8>, Ty::UnsignedVector<8>,
                 Ty::UnsignedVector<8>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.vsx.xvmaxsp",
     genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>>,
     genLibCall},
    {"__ppc_vec_max", "llvm.ppc.vsx.xvmaxdp",
     genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminsb",
     genFuncType<Ty::IntegerVector<1>, Ty::IntegerVector<1>,
                 Ty::IntegerVector<1>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminsh",
     genFuncType<Ty::IntegerVector<2>, Ty::IntegerVector<2>,
                 Ty::IntegerVector<2>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminsw",
     genFuncType<Ty::IntegerVector<4>, Ty::IntegerVector<4>,
                 Ty::IntegerVector<4>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminsd",
     genFuncType<Ty::IntegerVector<8>, Ty::IntegerVector<8>,
                 Ty::IntegerVector<8>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminub",
     genFuncType<Ty::UnsignedVector<1>, Ty::UnsignedVector<1>,
                 Ty::UnsignedVector<1>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminuh",
     genFuncType<Ty::UnsignedVector<2>, Ty::UnsignedVector<2>,
                 Ty::UnsignedVector<2>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminuw",
     genFuncType<Ty::UnsignedVector<4>, Ty::UnsignedVector<4>,
                 Ty::UnsignedVector<4>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.altivec.vminud",
     genFuncType<Ty::UnsignedVector<8>, Ty::UnsignedVector<8>,
                 Ty::UnsignedVector<8>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.vsx.xvminsp",
     genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>>,
     genLibCall},
    {"__ppc_vec_min", "llvm.ppc.vsx.xvmindp",
     genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>>,
     genLibCall},
    {"__ppc_vec_nmsub", "llvm.ppc.fnmsub.v4f32",
     genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>,
                 Ty::RealVector<4>>,
     genLibCall},
    {"__ppc_vec_nmsub", "llvm.ppc.fnmsub.v2f64",
     genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>,
                 Ty::RealVector<8>>,
     genLibCall},
};

const IntrinsicHandler *findPPCIntrinsicHandler(llvm::StringRef name) {
  auto compare = [](const IntrinsicHandler &ppcHandler, llvm::StringRef name) {
    return name.compare(ppcHandler.name) > 0;
  };
  auto result = llvm::lower_bound(ppcHandlers, name, compare);
  return result != std::end(ppcHandlers) && result->name == name ? result
                                                                 : nullptr;
}

using RtMap = Fortran::common::StaticMultimapView<MathOperation>;
static constexpr RtMap ppcMathOps(ppcMathOperations);
static_assert(ppcMathOps.Verify() && "map must be sorted");

std::pair<const MathOperation *, const MathOperation *>
checkPPCMathOperationsRange(llvm::StringRef name) {
  return ppcMathOps.equal_range(name);
}

static mlir::FunctionType genMmaVpFuncType(mlir::MLIRContext *context,
                                           int quadCnt, int pairCnt, int vecCnt,
                                           int intCnt = 0,
                                           int vecElemBitSize = 8,
                                           int intBitSize = 32) {
  // Constructs a function type with the following signature:
  // Result type: __vector_pair
  // Arguments:
  //   quadCnt: number of arguments that has __vector_quad type, followed by
  //   pairCnt: number of arguments that has __vector_pair type, followed by
  //   vecCnt: number of arguments that has vector(integer) type, followed by
  //   intCnt: number of arguments that has integer type
  //   vecElemBitSize: specifies the size of vector elements in bits
  //   intBitSize: specifies the size of integer arguments in bits
  auto vType{mlir::VectorType::get(
      128 / vecElemBitSize, mlir::IntegerType::get(context, vecElemBitSize))};
  auto vpType{fir::VectorType::get(256, mlir::IntegerType::get(context, 1))};
  auto vqType{fir::VectorType::get(512, mlir::IntegerType::get(context, 1))};
  auto iType{mlir::IntegerType::get(context, intBitSize)};
  llvm::SmallVector<mlir::Type> argTypes;
  for (int i = 0; i < quadCnt; ++i) {
    argTypes.push_back(vqType);
  }
  for (int i = 0; i < pairCnt; ++i) {
    argTypes.push_back(vpType);
  }
  for (int i = 0; i < vecCnt; ++i) {
    argTypes.push_back(vType);
  }
  for (int i = 0; i < intCnt; ++i) {
    argTypes.push_back(iType);
  }

  return mlir::FunctionType::get(context, argTypes, {vpType});
}

static mlir::FunctionType genMmaVqFuncType(mlir::MLIRContext *context,
                                           int quadCnt, int pairCnt, int vecCnt,
                                           int intCnt = 0,
                                           int vecElemBitSize = 8,
                                           int intBitSize = 32) {
  // Constructs a function type with the following signature:
  // Result type: __vector_quad
  // Arguments:
  //   quadCnt: number of arguments that has __vector_quad type, followed by
  //   pairCnt: number of arguments that has __vector_pair type, followed by
  //   vecCnt: number of arguments that has vector(integer) type, followed by
  //   intCnt: number of arguments that has integer type
  //   vecElemBitSize: specifies the size of vector elements in bits
  //   intBitSize: specifies the size of integer arguments in bits
  auto vType{mlir::VectorType::get(
      128 / vecElemBitSize, mlir::IntegerType::get(context, vecElemBitSize))};
  auto vpType{fir::VectorType::get(256, mlir::IntegerType::get(context, 1))};
  auto vqType{fir::VectorType::get(512, mlir::IntegerType::get(context, 1))};
  auto iType{mlir::IntegerType::get(context, intBitSize)};
  llvm::SmallVector<mlir::Type> argTypes;
  for (int i = 0; i < quadCnt; ++i) {
    argTypes.push_back(vqType);
  }
  for (int i = 0; i < pairCnt; ++i) {
    argTypes.push_back(vpType);
  }
  for (int i = 0; i < vecCnt; ++i) {
    argTypes.push_back(vType);
  }
  for (int i = 0; i < intCnt; ++i) {
    argTypes.push_back(iType);
  }

  return mlir::FunctionType::get(context, argTypes, {vqType});
}

mlir::FunctionType genMmaDisassembleFuncType(mlir::MLIRContext *context,
                                             MMAOp mmaOp) {
  auto vType{mlir::VectorType::get(16, mlir::IntegerType::get(context, 8))};
  llvm::SmallVector<mlir::Type> members;

  if (mmaOp == MMAOp::DisassembleAcc) {
    auto vqType{fir::VectorType::get(512, mlir::IntegerType::get(context, 1))};
    members.push_back(vType);
    members.push_back(vType);
    members.push_back(vType);
    members.push_back(vType);
    auto resType{mlir::LLVM::LLVMStructType::getLiteral(context, members)};
    return mlir::FunctionType::get(context, {vqType}, {resType});
  } else if (mmaOp == MMAOp::DisassemblePair) {
    auto vpType{fir::VectorType::get(256, mlir::IntegerType::get(context, 1))};
    members.push_back(vType);
    members.push_back(vType);
    auto resType{mlir::LLVM::LLVMStructType::getLiteral(context, members)};
    return mlir::FunctionType::get(context, {vpType}, {resType});
  } else {
    llvm_unreachable(
        "Unsupported intrinsic code for function signature generator");
  }
}

//===----------------------------------------------------------------------===//
// PowerPC specific intrinsic handlers.
//===----------------------------------------------------------------------===//

// MTFSF, MTFSFI
template <bool isImm>
void PPCIntrinsicLibrary::genMtfsf(llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  llvm::SmallVector<mlir::Value> scalarArgs;
  for (const fir::ExtendedValue &arg : args)
    if (arg.getUnboxed())
      scalarArgs.emplace_back(fir::getBase(arg));
    else
      mlir::emitError(loc, "nonscalar intrinsic argument");

  mlir::FunctionType libFuncType;
  mlir::func::FuncOp funcOp;
  if (isImm) {
    libFuncType = genFuncType<Ty::Void, Ty::Integer<4>, Ty::Integer<4>>(
        builder.getContext(), builder);
    funcOp = builder.addNamedFunction(loc, "llvm.ppc.mtfsfi", libFuncType);
  } else {
    libFuncType = genFuncType<Ty::Void, Ty::Integer<4>, Ty::Real<8>>(
        builder.getContext(), builder);
    funcOp = builder.addNamedFunction(loc, "llvm.ppc.mtfsf", libFuncType);
  }
  builder.create<fir::CallOp>(loc, funcOp, scalarArgs);
}

// VEC_ABS
fir::ExtendedValue
PPCIntrinsicLibrary::genVecAbs(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 1);
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto vTypeInfo{getVecTypeFromFir(argBases[0])};

  mlir::func::FuncOp funcOp{nullptr};
  mlir::FunctionType ftype;
  llvm::StringRef fname{};
  if (vTypeInfo.isFloat()) {
    if (vTypeInfo.isFloat32()) {
      fname = "llvm.fabs.v4f32";
      ftype =
          genFuncType<Ty::RealVector<4>, Ty::RealVector<4>>(context, builder);
    } else if (vTypeInfo.isFloat64()) {
      fname = "llvm.fabs.v2f64";
      ftype =
          genFuncType<Ty::RealVector<8>, Ty::RealVector<8>>(context, builder);
    }

    funcOp = builder.addNamedFunction(loc, fname, ftype);
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, argBases[0])};
    return callOp.getResult(0);
  } else if (auto eleTy = vTypeInfo.eleTy.dyn_cast<mlir::IntegerType>()) {
    // vec_abs(arg1) = max(0 - arg1, arg1)

    auto newVecTy{mlir::VectorType::get(vTypeInfo.len, eleTy)};
    auto varg1{builder.createConvert(loc, newVecTy, argBases[0])};
    // construct vector(0,..)
    auto zeroVal{builder.createIntegerConstant(loc, eleTy, 0)};
    auto vZero{
        builder.create<mlir::vector::BroadcastOp>(loc, newVecTy, zeroVal)};
    auto zeroSubVarg1{builder.create<mlir::arith::SubIOp>(loc, vZero, varg1)};

    mlir::func::FuncOp funcOp{nullptr};
    switch (eleTy.getWidth()) {
    case 8:
      fname = "llvm.ppc.altivec.vmaxsb";
      ftype = genFuncType<Ty::IntegerVector<1>, Ty::IntegerVector<1>,
                          Ty::IntegerVector<1>>(context, builder);
      break;
    case 16:
      fname = "llvm.ppc.altivec.vmaxsh";
      ftype = genFuncType<Ty::IntegerVector<2>, Ty::IntegerVector<2>,
                          Ty::IntegerVector<2>>(context, builder);
      break;
    case 32:
      fname = "llvm.ppc.altivec.vmaxsw";
      ftype = genFuncType<Ty::IntegerVector<4>, Ty::IntegerVector<4>,
                          Ty::IntegerVector<4>>(context, builder);
      break;
    case 64:
      fname = "llvm.ppc.altivec.vmaxsd";
      ftype = genFuncType<Ty::IntegerVector<8>, Ty::IntegerVector<8>,
                          Ty::IntegerVector<8>>(context, builder);
      break;
    default:
      llvm_unreachable("invalid integer size");
    }
    funcOp = builder.addNamedFunction(loc, fname, ftype);

    mlir::Value args[] = {zeroSubVarg1, varg1};
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, args)};
    return builder.createConvert(loc, argBases[0].getType(),
                                 callOp.getResult(0));
  }

  llvm_unreachable("unknown vector type");
}

// VEC_ADD, VEC_AND, VEC_SUB, VEC_MUL, VEC_XOR
template <VecOp vop>
fir::ExtendedValue PPCIntrinsicLibrary::genVecAddAndMulSubXor(
    mlir::Type resultType, llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  auto argBases{getBasesForArgs(args)};
  auto argsTy{getTypesForArgs(argBases)};
  assert(argsTy[0].isa<fir::VectorType>() && argsTy[1].isa<fir::VectorType>());

  auto vecTyInfo{getVecTypeFromFir(argBases[0])};

  const auto isInteger{vecTyInfo.eleTy.isa<mlir::IntegerType>()};
  const auto isFloat{vecTyInfo.eleTy.isa<mlir::FloatType>()};
  assert((isInteger || isFloat) && "unknown vector type");

  auto vargs{convertVecArgs(builder, loc, vecTyInfo, argBases)};

  mlir::Value r{nullptr};
  switch (vop) {
  case VecOp::Add:
    if (isInteger)
      r = builder.create<mlir::arith::AddIOp>(loc, vargs[0], vargs[1]);
    else if (isFloat)
      r = builder.create<mlir::arith::AddFOp>(loc, vargs[0], vargs[1]);
    break;
  case VecOp::Mul:
    if (isInteger)
      r = builder.create<mlir::arith::MulIOp>(loc, vargs[0], vargs[1]);
    else if (isFloat)
      r = builder.create<mlir::arith::MulFOp>(loc, vargs[0], vargs[1]);
    break;
  case VecOp::Sub:
    if (isInteger)
      r = builder.create<mlir::arith::SubIOp>(loc, vargs[0], vargs[1]);
    else if (isFloat)
      r = builder.create<mlir::arith::SubFOp>(loc, vargs[0], vargs[1]);
    break;
  case VecOp::And:
  case VecOp::Xor: {
    mlir::Value arg1{nullptr};
    mlir::Value arg2{nullptr};
    if (isInteger) {
      arg1 = vargs[0];
      arg2 = vargs[1];
    } else if (isFloat) {
      // bitcast the arguments to integer
      auto wd{vecTyInfo.eleTy.dyn_cast<mlir::FloatType>().getWidth()};
      auto ftype{builder.getIntegerType(wd)};
      auto bcVecTy{mlir::VectorType::get(vecTyInfo.len, ftype)};
      arg1 = builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[0]);
      arg2 = builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[1]);
    }
    if (vop == VecOp::And)
      r = builder.create<mlir::arith::AndIOp>(loc, arg1, arg2);
    else if (vop == VecOp::Xor)
      r = builder.create<mlir::arith::XOrIOp>(loc, arg1, arg2);

    if (isFloat)
      r = builder.create<mlir::vector::BitCastOp>(loc, vargs[0].getType(), r);

    break;
  }
  }

  return builder.createConvert(loc, argsTy[0], r);
}

// VEC_ANY_GE
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecAnyCompare(mlir::Type resultType,
                                      llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  assert(vop == VecOp::Anyge && "unknown vector compare operation");
  auto argBases{getBasesForArgs(args)};
  VecTypeInfo vTypeInfo{getVecTypeFromFir(argBases[0])};
  [[maybe_unused]] const auto isSupportedTy{
      mlir::isa<mlir::Float32Type, mlir::Float64Type, mlir::IntegerType>(
          vTypeInfo.eleTy)};
  assert(isSupportedTy && "unsupported vector type");

  // Constants for mapping CR6 bits to predicate result
  enum { CR6_EQ_REV = 1, CR6_LT_REV = 3 };

  auto context{builder.getContext()};

  static std::map<std::pair<ParamTypeId, unsigned>,
                  std::pair<llvm::StringRef, mlir::FunctionType>>
      uiBuiltin{
          {std::make_pair(ParamTypeId::IntegerVector, 8),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsb.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::IntegerVector<1>,
                           Ty::IntegerVector<1>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 16),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsh.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::IntegerVector<2>,
                           Ty::IntegerVector<2>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 32),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsw.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::IntegerVector<4>,
                           Ty::IntegerVector<4>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 64),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsd.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::IntegerVector<8>,
                           Ty::IntegerVector<8>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 8),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtub.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>,
                           Ty::UnsignedVector<1>, Ty::UnsignedVector<1>>(
                   context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 16),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtuh.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>,
                           Ty::UnsignedVector<2>, Ty::UnsignedVector<2>>(
                   context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 32),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtuw.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>,
                           Ty::UnsignedVector<4>, Ty::UnsignedVector<4>>(
                   context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 64),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtud.p",
               genFuncType<Ty::Integer<4>, Ty::Integer<4>,
                           Ty::UnsignedVector<8>, Ty::UnsignedVector<8>>(
                   context, builder))},
      };

  mlir::FunctionType ftype{nullptr};
  llvm::StringRef fname;
  const auto i32Ty{mlir::IntegerType::get(context, 32)};
  llvm::SmallVector<mlir::Value> cmpArgs;
  mlir::Value op{nullptr};
  const auto width{vTypeInfo.eleTy.getIntOrFloatBitWidth()};

  if (auto elementTy = mlir::dyn_cast<mlir::IntegerType>(vTypeInfo.eleTy)) {
    std::pair<llvm::StringRef, mlir::FunctionType> bi;
    bi = (elementTy.isUnsignedInteger())
             ? uiBuiltin[std::pair(ParamTypeId::UnsignedVector, width)]
             : uiBuiltin[std::pair(ParamTypeId::IntegerVector, width)];

    fname = std::get<0>(bi);
    ftype = std::get<1>(bi);

    op = builder.createIntegerConstant(loc, i32Ty, CR6_LT_REV);
    cmpArgs.emplace_back(op);
    // reverse the argument order
    cmpArgs.emplace_back(argBases[1]);
    cmpArgs.emplace_back(argBases[0]);
  } else if (vTypeInfo.isFloat()) {
    if (vTypeInfo.isFloat32()) {
      fname = "llvm.ppc.vsx.xvcmpgesp.p";
      ftype = genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::RealVector<4>,
                          Ty::RealVector<4>>(context, builder);
    } else {
      fname = "llvm.ppc.vsx.xvcmpgedp.p";
      ftype = genFuncType<Ty::Integer<4>, Ty::Integer<4>, Ty::RealVector<8>,
                          Ty::RealVector<8>>(context, builder);
    }
    op = builder.createIntegerConstant(loc, i32Ty, CR6_EQ_REV);
    cmpArgs.emplace_back(op);
    cmpArgs.emplace_back(argBases[0]);
    cmpArgs.emplace_back(argBases[1]);
  }
  assert((!fname.empty() && ftype) && "invalid type");

  mlir::func::FuncOp funcOp{builder.addNamedFunction(loc, fname, ftype)};
  auto callOp{builder.create<fir::CallOp>(loc, funcOp, cmpArgs)};
  return callOp.getResult(0);
}

static std::pair<llvm::StringRef, mlir::FunctionType>
getVecCmpFuncTypeAndName(VecTypeInfo &vTypeInfo, VecOp vop,
                         fir::FirOpBuilder &builder) {
  auto context{builder.getContext()};
  static std::map<std::pair<ParamTypeId, unsigned>,
                  std::pair<llvm::StringRef, mlir::FunctionType>>
      iuBuiltinName{
          {std::make_pair(ParamTypeId::IntegerVector, 8),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsb",
               genFuncType<Ty::UnsignedVector<1>, Ty::IntegerVector<1>,
                           Ty::IntegerVector<1>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 16),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsh",
               genFuncType<Ty::UnsignedVector<2>, Ty::IntegerVector<2>,
                           Ty::IntegerVector<2>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 32),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsw",
               genFuncType<Ty::UnsignedVector<4>, Ty::IntegerVector<4>,
                           Ty::IntegerVector<4>>(context, builder))},
          {std::make_pair(ParamTypeId::IntegerVector, 64),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtsd",
               genFuncType<Ty::UnsignedVector<8>, Ty::IntegerVector<8>,
                           Ty::IntegerVector<8>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 8),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtub",
               genFuncType<Ty::UnsignedVector<1>, Ty::UnsignedVector<1>,
                           Ty::UnsignedVector<1>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 16),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtuh",
               genFuncType<Ty::UnsignedVector<2>, Ty::UnsignedVector<2>,
                           Ty::UnsignedVector<2>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 32),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtuw",
               genFuncType<Ty::UnsignedVector<4>, Ty::UnsignedVector<4>,
                           Ty::UnsignedVector<4>>(context, builder))},
          {std::make_pair(ParamTypeId::UnsignedVector, 64),
           std::make_pair(
               "llvm.ppc.altivec.vcmpgtud",
               genFuncType<Ty::UnsignedVector<8>, Ty::UnsignedVector<8>,
                           Ty::UnsignedVector<8>>(context, builder))}};

  // VSX only defines GE and GT builtins. Cmple and Cmplt use GE and GT with
  // arguments revsered.
  enum class Cmp { gtOrLt, geOrLe };
  static std::map<std::pair<Cmp, int>,
                  std::pair<llvm::StringRef, mlir::FunctionType>>
      rGBI{{std::make_pair(Cmp::geOrLe, 32),
            std::make_pair("llvm.ppc.vsx.xvcmpgesp",
                           genFuncType<Ty::UnsignedVector<4>, Ty::RealVector<4>,
                                       Ty::RealVector<4>>(context, builder))},
           {std::make_pair(Cmp::geOrLe, 64),
            std::make_pair("llvm.ppc.vsx.xvcmpgedp",
                           genFuncType<Ty::UnsignedVector<8>, Ty::RealVector<8>,
                                       Ty::RealVector<8>>(context, builder))},
           {std::make_pair(Cmp::gtOrLt, 32),
            std::make_pair("llvm.ppc.vsx.xvcmpgtsp",
                           genFuncType<Ty::UnsignedVector<4>, Ty::RealVector<4>,
                                       Ty::RealVector<4>>(context, builder))},
           {std::make_pair(Cmp::gtOrLt, 64),
            std::make_pair("llvm.ppc.vsx.xvcmpgtdp",
                           genFuncType<Ty::UnsignedVector<8>, Ty::RealVector<8>,
                                       Ty::RealVector<8>>(context, builder))}};

  const auto width{vTypeInfo.eleTy.getIntOrFloatBitWidth()};
  std::pair<llvm::StringRef, mlir::FunctionType> specFunc;
  if (auto elementTy = mlir::dyn_cast<mlir::IntegerType>(vTypeInfo.eleTy))
    specFunc =
        (elementTy.isUnsignedInteger())
            ? iuBuiltinName[std::make_pair(ParamTypeId::UnsignedVector, width)]
            : iuBuiltinName[std::make_pair(ParamTypeId::IntegerVector, width)];
  else if (vTypeInfo.isFloat())
    specFunc = (vop == VecOp::Cmpge || vop == VecOp::Cmple)
                   ? rGBI[std::make_pair(Cmp::geOrLe, width)]
                   : rGBI[std::make_pair(Cmp::gtOrLt, width)];

  assert(!std::get<0>(specFunc).empty() && "unknown builtin name");
  assert(std::get<1>(specFunc) && "unknown function type");
  return specFunc;
}

// VEC_CMPGE, VEC_CMPGT, VEC_CMPLE, VEC_CMPLT
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecCmp(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 2);
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  VecTypeInfo vecTyInfo{getVecTypeFromFir(argBases[0])};
  auto varg{convertVecArgs(builder, loc, vecTyInfo, argBases)};

  std::pair<llvm::StringRef, mlir::FunctionType> funcTyNam{
      getVecCmpFuncTypeAndName(vecTyInfo, vop, builder)};

  mlir::func::FuncOp funcOp = builder.addNamedFunction(
      loc, std::get<0>(funcTyNam), std::get<1>(funcTyNam));

  mlir::Value res{nullptr};

  if (auto eTy = vecTyInfo.eleTy.dyn_cast<mlir::IntegerType>()) {
    constexpr int firstArg{0};
    constexpr int secondArg{1};
    std::map<VecOp, std::array<int, 2>> argOrder{
        {VecOp::Cmpge, {secondArg, firstArg}},
        {VecOp::Cmple, {firstArg, secondArg}},
        {VecOp::Cmpgt, {firstArg, secondArg}},
        {VecOp::Cmplt, {secondArg, firstArg}}};

    // Construct the function return type, unsigned vector, for conversion.
    auto itype = mlir::IntegerType::get(context, eTy.getWidth(),
                                        mlir::IntegerType::Unsigned);
    auto returnType = fir::VectorType::get(vecTyInfo.len, itype);

    switch (vop) {
    case VecOp::Cmpgt:
    case VecOp::Cmplt: {
      // arg1 > arg2 --> vcmpgt(arg1, arg2)
      // arg1 < arg2 --> vcmpgt(arg2, arg1)
      mlir::Value vargs[]{argBases[argOrder[vop][0]],
                          argBases[argOrder[vop][1]]};
      auto callOp{builder.create<fir::CallOp>(loc, funcOp, vargs)};
      res = callOp.getResult(0);
      break;
    }
    case VecOp::Cmpge:
    case VecOp::Cmple: {
      // arg1 >= arg2 --> vcmpge(arg2, arg1) xor vector(-1)
      // arg1 <= arg2 --> vcmpge(arg1, arg2) xor vector(-1)
      mlir::Value vargs[]{argBases[argOrder[vop][0]],
                          argBases[argOrder[vop][1]]};

      // Construct a constant vector(-1)
      auto negOneVal{builder.createIntegerConstant(
          loc, getConvertedElementType(context, eTy), -1)};
      auto vNegOne{builder.create<mlir::vector::BroadcastOp>(
          loc, vecTyInfo.toMlirVectorType(context), negOneVal)};

      auto callOp{builder.create<fir::CallOp>(loc, funcOp, vargs)};
      mlir::Value callRes{callOp.getResult(0)};
      auto vargs2{
          convertVecArgs(builder, loc, vecTyInfo, mlir::ValueRange{callRes})};
      auto xorRes{builder.create<mlir::arith::XOrIOp>(loc, vargs2[0], vNegOne)};

      res = builder.createConvert(loc, returnType, xorRes);
      break;
    }
    default:
      llvm_unreachable("Invalid vector operation for generator");
    }
  } else if (vecTyInfo.isFloat()) {
    mlir::Value vargs[2];
    switch (vop) {
    case VecOp::Cmpge:
    case VecOp::Cmpgt:
      vargs[0] = argBases[0];
      vargs[1] = argBases[1];
      break;
    case VecOp::Cmple:
    case VecOp::Cmplt:
      // Swap the arguments as xvcmpg[et] is used
      vargs[0] = argBases[1];
      vargs[1] = argBases[0];
      break;
    default:
      llvm_unreachable("Invalid vector operation for generator");
    }
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, vargs)};
    res = callOp.getResult(0);
  } else
    llvm_unreachable("invalid vector type");

  return res;
}

static inline mlir::Value swapVectorWordPairs(fir::FirOpBuilder &builder,
                                              mlir::Location loc,
                                              mlir::Value arg) {
  auto ty = arg.getType();
  auto context{builder.getContext()};
  auto vtype{mlir::VectorType::get(16, mlir::IntegerType::get(context, 8))};

  if (ty != vtype)
    arg = builder.create<mlir::LLVM::BitcastOp>(loc, vtype, arg).getResult();

  llvm::SmallVector<int64_t, 16> mask{4,  5,  6,  7,  0, 1, 2,  3,
                                      12, 13, 14, 15, 8, 9, 10, 11};
  arg = builder.create<mlir::vector::ShuffleOp>(loc, arg, arg, mask);
  if (ty != vtype)
    arg = builder.create<mlir::LLVM::BitcastOp>(loc, ty, arg);
  return arg;
}

// VEC_CONVERT, VEC_CTF, VEC_CVF
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecConvert(mlir::Type resultType,
                                   llvm::ArrayRef<fir::ExtendedValue> args) {
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto vecTyInfo{getVecTypeFromFir(argBases[0])};
  auto mlirTy{vecTyInfo.toMlirVectorType(context)};
  auto vArg1{builder.createConvert(loc, mlirTy, argBases[0])};
  const auto i32Ty{mlir::IntegerType::get(context, 32)};

  switch (vop) {
  case VecOp::Ctf: {
    assert(args.size() == 2);
    auto convArg{builder.createConvert(loc, i32Ty, argBases[1])};
    auto eTy{vecTyInfo.eleTy.dyn_cast<mlir::IntegerType>()};
    assert(eTy && "Unsupported vector type");
    const auto isUnsigned{eTy.isUnsignedInteger()};
    const auto width{eTy.getWidth()};

    if (width == 32) {
      auto ftype{(isUnsigned)
                     ? genFuncType<Ty::RealVector<4>, Ty::UnsignedVector<4>,
                                   Ty::Integer<4>>(context, builder)
                     : genFuncType<Ty::RealVector<4>, Ty::IntegerVector<4>,
                                   Ty::Integer<4>>(context, builder)};
      const llvm::StringRef fname{(isUnsigned) ? "llvm.ppc.altivec.vcfux"
                                               : "llvm.ppc.altivec.vcfsx"};
      auto funcOp{builder.addNamedFunction(loc, fname, ftype)};
      mlir::Value newArgs[] = {argBases[0], convArg};
      auto callOp{builder.create<fir::CallOp>(loc, funcOp, newArgs)};

      return callOp.getResult(0);
    } else if (width == 64) {
      auto fTy{mlir::FloatType::getF64(context)};
      auto ty{mlir::VectorType::get(2, fTy)};

      // vec_vtf(arg1, arg2) = fmul(1.0 / (1 << arg2), llvm.sitofp(arg1))
      auto convOp{(isUnsigned)
                      ? builder.create<mlir::LLVM::UIToFPOp>(loc, ty, vArg1)
                      : builder.create<mlir::LLVM::SIToFPOp>(loc, ty, vArg1)};

      // construct vector<1./(1<<arg1), 1.0/(1<<arg1)>
      auto constInt{
          mlir::dyn_cast<mlir::arith::ConstantOp>(argBases[1].getDefiningOp())
              .getValue()
              .dyn_cast_or_null<mlir::IntegerAttr>()};
      assert(constInt && "expected integer constant argument");
      double f{1.0 / (1 << constInt.getInt())};
      llvm::SmallVector<double> vals{f, f};
      auto constOp{builder.create<mlir::arith::ConstantOp>(
          loc, ty, builder.getF64VectorAttr(vals))};

      auto mulOp{builder.create<mlir::LLVM::FMulOp>(
          loc, ty, convOp->getResult(0), constOp)};

      return builder.createConvert(loc, fir::VectorType::get(2, fTy), mulOp);
    }
    llvm_unreachable("invalid element integer kind");
  }
  case VecOp::Convert: {
    assert(args.size() == 2);
    // resultType has mold type (if scalar) or element type (if array)
    auto resTyInfo{getVecTypeFromFirType(resultType)};
    auto moldTy{resTyInfo.toMlirVectorType(context)};
    auto firTy{resTyInfo.toFirVectorType()};

    // vec_convert(v, mold) = bitcast v to "type of mold"
    auto conv{builder.create<mlir::LLVM::BitcastOp>(loc, moldTy, vArg1)};

    return builder.createConvert(loc, firTy, conv);
  }
  case VecOp::Cvf: {
    assert(args.size() == 1);

    mlir::Value newArgs[]{vArg1};
    if (vecTyInfo.isFloat32()) {
      // TODO: Handle element ordering
      newArgs[0] = swapVectorWordPairs(builder, loc, newArgs[0]);

      const llvm::StringRef fname{"llvm.ppc.vsx.xvcvspdp"};
      auto ftype{
          genFuncType<Ty::RealVector<8>, Ty::RealVector<4>>(context, builder)};
      auto funcOp{builder.addNamedFunction(loc, fname, ftype)};
      auto callOp{builder.create<fir::CallOp>(loc, funcOp, newArgs)};

      return callOp.getResult(0);
    } else if (vecTyInfo.isFloat64()) {
      const llvm::StringRef fname{"llvm.ppc.vsx.xvcvdpsp"};
      auto ftype{
          genFuncType<Ty::RealVector<4>, Ty::RealVector<8>>(context, builder)};
      auto funcOp{builder.addNamedFunction(loc, fname, ftype)};
      newArgs[0] =
          builder.create<fir::CallOp>(loc, funcOp, newArgs).getResult(0);
      auto fvf32Ty{newArgs[0].getType()};
      auto f32type{mlir::FloatType::getF32(context)};
      auto mvf32Ty{mlir::VectorType::get(4, f32type)};
      newArgs[0] = builder.createConvert(loc, mvf32Ty, newArgs[0]);

      // TODO: Handle element ordering
      newArgs[0] = swapVectorWordPairs(builder, loc, newArgs[0]);

      return builder.createConvert(loc, fvf32Ty, newArgs[0]);
    }
    llvm_unreachable("invalid element integer kind");
  }
  default:
    llvm_unreachable("Invalid vector operation for generator");
  }
}

// VEC_NMADD, VEC_MSUB
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecNmaddMsub(mlir::Type resultType,
                                     llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto vTypeInfo{getVecTypeFromFir(argBases[0])};
  auto newArgs{convertVecArgs(builder, loc, vTypeInfo, argBases)};
  const auto width{vTypeInfo.eleTy.getIntOrFloatBitWidth()};

  static std::map<int, std::pair<llvm::StringRef, mlir::FunctionType>> fmaMap{
      {32,
       std::make_pair(
           "llvm.fma.v4f32",
           genFuncType<Ty::RealVector<4>, Ty::RealVector<4>, Ty::RealVector<4>>(
               context, builder))},
      {64,
       std::make_pair(
           "llvm.fma.v2f64",
           genFuncType<Ty::RealVector<8>, Ty::RealVector<8>, Ty::RealVector<8>>(
               context, builder))}};

  auto funcOp{builder.addNamedFunction(loc, std::get<0>(fmaMap[width]),
                                       std::get<1>(fmaMap[width]))};
  if (vop == VecOp::Nmadd) {
    // vec_nmadd(arg1, arg2, arg3) = -fma(arg1, arg2, arg3)
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, newArgs)};

    // We need to convert fir.vector to MLIR vector to use fneg and then back
    // to fir.vector to store.
    auto vCall{builder.createConvert(loc, vTypeInfo.toMlirVectorType(context),
                                     callOp.getResult(0))};
    auto neg{builder.create<mlir::arith::NegFOp>(loc, vCall)};
    return builder.createConvert(loc, vTypeInfo.toFirVectorType(), neg);
  } else if (vop == VecOp::Msub) {
    // vec_msub(arg1, arg2, arg3) = fma(arg1, arg2, -arg3)
    newArgs[2] = builder.create<mlir::arith::NegFOp>(loc, newArgs[2]);

    auto callOp{builder.create<fir::CallOp>(loc, funcOp, newArgs)};
    return callOp.getResult(0);
  }
  llvm_unreachable("Invalid vector operation for generator");
}

// VEC_SEL
fir::ExtendedValue
PPCIntrinsicLibrary::genVecSel(mlir::Type resultType,
                               llvm::ArrayRef<fir::ExtendedValue> args) {
  assert(args.size() == 3);
  auto argBases{getBasesForArgs(args)};
  llvm::SmallVector<VecTypeInfo, 4> vecTyInfos;
  for (size_t i = 0; i < argBases.size(); i++) {
    vecTyInfos.push_back(getVecTypeFromFir(argBases[i]));
  }
  auto vargs{convertVecArgs(builder, loc, vecTyInfos, argBases)};

  auto i8Ty{mlir::IntegerType::get(builder.getContext(), 8)};
  auto negOne{builder.createIntegerConstant(loc, i8Ty, -1)};

  // construct a constant <16 x i8> vector with value -1 for bitcast
  auto bcVecTy{mlir::VectorType::get(16, i8Ty)};
  auto vNegOne{builder.create<mlir::vector::BroadcastOp>(loc, bcVecTy, negOne)};

  // bitcast arguments to bcVecTy
  auto arg1{builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[0])};
  auto arg2{builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[1])};
  auto arg3{builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, vargs[2])};

  // vec_sel(arg1, arg2, arg3) =
  //   (arg2 and arg3) or (arg1 and (arg3 xor vector(-1,...)))
  auto comp{builder.create<mlir::arith::XOrIOp>(loc, arg3, vNegOne)};
  auto a1AndComp{builder.create<mlir::arith::AndIOp>(loc, arg1, comp)};
  auto a1OrA2{builder.create<mlir::arith::AndIOp>(loc, arg2, arg3)};
  auto res{builder.create<mlir::arith::OrIOp>(loc, a1AndComp, a1OrA2)};

  auto bcRes{
      builder.create<mlir::vector::BitCastOp>(loc, vargs[0].getType(), res)};

  return builder.createConvert(loc, vecTyInfos[0].toFirVectorType(), bcRes);
}

// VEC_SL, VEC_SLD, VEC_SLDW, VEC_SLL, VEC_SLO, VEC_SR, VEC_SRL, VEC_SRO
template <VecOp vop>
fir::ExtendedValue
PPCIntrinsicLibrary::genVecShift(mlir::Type resultType,
                                 llvm::ArrayRef<fir::ExtendedValue> args) {
  auto context{builder.getContext()};
  auto argBases{getBasesForArgs(args)};
  auto argTypes{getTypesForArgs(argBases)};

  llvm::SmallVector<VecTypeInfo, 2> vecTyInfoArgs;
  vecTyInfoArgs.push_back(getVecTypeFromFir(argBases[0]));
  vecTyInfoArgs.push_back(getVecTypeFromFir(argBases[1]));

  // Convert the first two arguments to MLIR vectors
  llvm::SmallVector<mlir::Type, 2> mlirTyArgs;
  mlirTyArgs.push_back(vecTyInfoArgs[0].toMlirVectorType(context));
  mlirTyArgs.push_back(vecTyInfoArgs[1].toMlirVectorType(context));

  llvm::SmallVector<mlir::Value, 2> mlirVecArgs;
  mlirVecArgs.push_back(builder.createConvert(loc, mlirTyArgs[0], argBases[0]));
  mlirVecArgs.push_back(builder.createConvert(loc, mlirTyArgs[1], argBases[1]));

  mlir::Value shftRes{nullptr};

  if (vop == VecOp::Sl || vop == VecOp::Sr) {
    assert(args.size() == 2);
    // Construct the mask
    auto width{
        mlir::dyn_cast<mlir::IntegerType>(vecTyInfoArgs[1].eleTy).getWidth()};
    auto vecVal{builder.createIntegerConstant(
        loc, getConvertedElementType(context, vecTyInfoArgs[0].eleTy), width)};
    auto mask{
        builder.create<mlir::vector::BroadcastOp>(loc, mlirTyArgs[1], vecVal)};
    auto shft{builder.create<mlir::arith::RemUIOp>(loc, mlirVecArgs[1], mask)};

    mlir::Value res{nullptr};
    if (vop == VecOp::Sr)
      res = builder.create<mlir::arith::ShRUIOp>(loc, mlirVecArgs[0], shft);
    else if (vop == VecOp::Sl)
      res = builder.create<mlir::arith::ShLIOp>(loc, mlirVecArgs[0], shft);

    shftRes = builder.createConvert(loc, argTypes[0], res);
  } else if (vop == VecOp::Sll || vop == VecOp::Slo || vop == VecOp::Srl ||
             vop == VecOp::Sro) {
    assert(args.size() == 2);

    // Bitcast to vector<4xi32>
    auto bcVecTy{mlir::VectorType::get(4, builder.getIntegerType(32))};
    if (mlirTyArgs[0] != bcVecTy)
      mlirVecArgs[0] =
          builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, mlirVecArgs[0]);
    if (mlirTyArgs[1] != bcVecTy)
      mlirVecArgs[1] =
          builder.create<mlir::vector::BitCastOp>(loc, bcVecTy, mlirVecArgs[1]);

    llvm::StringRef funcName;
    switch (vop) {
    case VecOp::Srl:
      funcName = "llvm.ppc.altivec.vsr";
      break;
    case VecOp::Sro:
      funcName = "llvm.ppc.altivec.vsro";
      break;
    case VecOp::Sll:
      funcName = "llvm.ppc.altivec.vsl";
      break;
    case VecOp::Slo:
      funcName = "llvm.ppc.altivec.vslo";
      break;
    default:
      llvm_unreachable("unknown vector shift operation");
    }
    auto funcTy{genFuncType<Ty::IntegerVector<4>, Ty::IntegerVector<4>,
                            Ty::IntegerVector<4>>(context, builder)};
    mlir::func::FuncOp funcOp{builder.addNamedFunction(loc, funcName, funcTy)};
    auto callOp{builder.create<fir::CallOp>(loc, funcOp, mlirVecArgs)};

    // If the result vector type is different from the original type, need
    // to convert to mlir vector, bitcast and then convert back to fir vector.
    if (callOp.getResult(0).getType() != argTypes[0]) {
      auto res = builder.createConvert(loc, bcVecTy, callOp.getResult(0));
      res = builder.create<mlir::vector::BitCastOp>(loc, mlirTyArgs[0], res);
      shftRes = builder.createConvert(loc, argTypes[0], res);
    } else {
      shftRes = callOp.getResult(0);
    }
  } else if (vop == VecOp::Sld || vop == VecOp::Sldw) {
    assert(args.size() == 3);
    auto constIntOp =
        mlir::dyn_cast<mlir::arith::ConstantOp>(argBases[2].getDefiningOp())
            .getValue()
            .dyn_cast_or_null<mlir::IntegerAttr>();
    assert(constIntOp && "expected integer constant argument");

    // Bitcast to vector<16xi8>
    auto vi8Ty{mlir::VectorType::get(16, builder.getIntegerType(8))};
    if (mlirTyArgs[0] != vi8Ty) {
      mlirVecArgs[0] =
          builder.create<mlir::LLVM::BitcastOp>(loc, vi8Ty, mlirVecArgs[0])
              .getResult();
      mlirVecArgs[1] =
          builder.create<mlir::LLVM::BitcastOp>(loc, vi8Ty, mlirVecArgs[1])
              .getResult();
    }

    // Construct the mask for shuffling
    auto shiftVal{constIntOp.getInt()};
    if (vop == VecOp::Sldw)
      shiftVal = shiftVal << 2;
    shiftVal &= 0xF;
    llvm::SmallVector<int64_t, 16> mask;
    for (int i = 16; i < 32; ++i)
      mask.push_back(i - shiftVal);

    // Shuffle with mask
    shftRes = builder.create<mlir::vector::ShuffleOp>(loc, mlirVecArgs[1],
                                                      mlirVecArgs[0], mask);

    // Bitcast to the original type
    if (shftRes.getType() != mlirTyArgs[0])
      shftRes =
          builder.create<mlir::LLVM::BitcastOp>(loc, mlirTyArgs[0], shftRes);

    return builder.createConvert(loc, resultType, shftRes);
  } else
    llvm_unreachable("Invalid vector operation for generator");

  return shftRes;
}

const char *getMmaIrIntrName(MMAOp mmaOp) {
  switch (mmaOp) {
  case MMAOp::AssembleAcc:
    return "llvm.ppc.mma.assemble.acc";
  case MMAOp::AssemblePair:
    return "llvm.ppc.vsx.assemble.pair";
  case MMAOp::DisassembleAcc:
    return "llvm.ppc.mma.disassemble.acc";
  case MMAOp::DisassemblePair:
    return "llvm.ppc.vsx.disassemble.pair";
  case MMAOp::Xxmfacc:
    return "llvm.ppc.mma.xxmfacc";
  case MMAOp::Xxmtacc:
    return "llvm.ppc.mma.xxmtacc";
  case MMAOp::Xxsetaccz:
    return "llvm.ppc.mma.xxsetaccz";
  case MMAOp::Pmxvbf16ger2:
    return "llvm.ppc.mma.pmxvbf16ger2";
  case MMAOp::Pmxvbf16ger2nn:
    return "llvm.ppc.mma.pmxvbf16ger2nn";
  case MMAOp::Pmxvbf16ger2np:
    return "llvm.ppc.mma.pmxvbf16ger2np";
  case MMAOp::Pmxvbf16ger2pn:
    return "llvm.ppc.mma.pmxvbf16ger2pn";
  case MMAOp::Pmxvbf16ger2pp:
    return "llvm.ppc.mma.pmxvbf16ger2pp";
  case MMAOp::Pmxvf16ger2:
    return "llvm.ppc.mma.pmxvf16ger2";
  case MMAOp::Pmxvf16ger2nn:
    return "llvm.ppc.mma.pmxvf16ger2nn";
  case MMAOp::Pmxvf16ger2np:
    return "llvm.ppc.mma.pmxvf16ger2np";
  case MMAOp::Pmxvf16ger2pn:
    return "llvm.ppc.mma.pmxvf16ger2pn";
  case MMAOp::Pmxvf16ger2pp:
    return "llvm.ppc.mma.pmxvf16ger2pp";
  case MMAOp::Pmxvf32ger:
    return "llvm.ppc.mma.pmxvf32ger";
  case MMAOp::Pmxvf32gernn:
    return "llvm.ppc.mma.pmxvf32gernn";
  case MMAOp::Pmxvf32gernp:
    return "llvm.ppc.mma.pmxvf32gernp";
  case MMAOp::Pmxvf32gerpn:
    return "llvm.ppc.mma.pmxvf32gerpn";
  case MMAOp::Pmxvf32gerpp:
    return "llvm.ppc.mma.pmxvf32gerpp";
  case MMAOp::Pmxvf64ger:
    return "llvm.ppc.mma.pmxvf64ger";
  case MMAOp::Pmxvf64gernn:
    return "llvm.ppc.mma.pmxvf64gernn";
  case MMAOp::Pmxvf64gernp:
    return "llvm.ppc.mma.pmxvf64gernp";
  case MMAOp::Pmxvf64gerpn:
    return "llvm.ppc.mma.pmxvf64gerpn";
  case MMAOp::Pmxvf64gerpp:
    return "llvm.ppc.mma.pmxvf64gerpp";
  case MMAOp::Pmxvi16ger2:
    return "llvm.ppc.mma.pmxvi16ger2";
  case MMAOp::Pmxvi16ger2pp:
    return "llvm.ppc.mma.pmxvi16ger2pp";
  case MMAOp::Pmxvi16ger2s:
    return "llvm.ppc.mma.pmxvi16ger2s";
  case MMAOp::Pmxvi16ger2spp:
    return "llvm.ppc.mma.pmxvi16ger2spp";
  case MMAOp::Pmxvi4ger8:
    return "llvm.ppc.mma.pmxvi4ger8";
  case MMAOp::Pmxvi4ger8pp:
    return "llvm.ppc.mma.pmxvi4ger8pp";
  case MMAOp::Pmxvi8ger4:
    return "llvm.ppc.mma.pmxvi8ger4";
  case MMAOp::Pmxvi8ger4pp:
    return "llvm.ppc.mma.pmxvi8ger4pp";
  case MMAOp::Pmxvi8ger4spp:
    return "llvm.ppc.mma.pmxvi8ger4spp";
  case MMAOp::Xvbf16ger2:
    return "llvm.ppc.mma.xvbf16ger2";
  case MMAOp::Xvbf16ger2nn:
    return "llvm.ppc.mma.xvbf16ger2nn";
  case MMAOp::Xvbf16ger2np:
    return "llvm.ppc.mma.xvbf16ger2np";
  case MMAOp::Xvbf16ger2pn:
    return "llvm.ppc.mma.xvbf16ger2pn";
  case MMAOp::Xvbf16ger2pp:
    return "llvm.ppc.mma.xvbf16ger2pp";
  case MMAOp::Xvf16ger2:
    return "llvm.ppc.mma.xvf16ger2";
  case MMAOp::Xvf16ger2nn:
    return "llvm.ppc.mma.xvf16ger2nn";
  case MMAOp::Xvf16ger2np:
    return "llvm.ppc.mma.xvf16ger2np";
  case MMAOp::Xvf16ger2pn:
    return "llvm.ppc.mma.xvf16ger2pn";
  case MMAOp::Xvf16ger2pp:
    return "llvm.ppc.mma.xvf16ger2pp";
  case MMAOp::Xvf32ger:
    return "llvm.ppc.mma.xvf32ger";
  case MMAOp::Xvf32gernn:
    return "llvm.ppc.mma.xvf32gernn";
  case MMAOp::Xvf32gernp:
    return "llvm.ppc.mma.xvf32gernp";
  case MMAOp::Xvf32gerpn:
    return "llvm.ppc.mma.xvf32gerpn";
  case MMAOp::Xvf32gerpp:
    return "llvm.ppc.mma.xvf32gerpp";
  case MMAOp::Xvf64ger:
    return "llvm.ppc.mma.xvf64ger";
  case MMAOp::Xvf64gernn:
    return "llvm.ppc.mma.xvf64gernn";
  case MMAOp::Xvf64gernp:
    return "llvm.ppc.mma.xvf64gernp";
  case MMAOp::Xvf64gerpn:
    return "llvm.ppc.mma.xvf64gerpn";
  case MMAOp::Xvf64gerpp:
    return "llvm.ppc.mma.xvf64gerpp";
  case MMAOp::Xvi16ger2:
    return "llvm.ppc.mma.xvi16ger2";
  case MMAOp::Xvi16ger2pp:
    return "llvm.ppc.mma.xvi16ger2pp";
  case MMAOp::Xvi16ger2s:
    return "llvm.ppc.mma.xvi16ger2s";
  case MMAOp::Xvi16ger2spp:
    return "llvm.ppc.mma.xvi16ger2spp";
  case MMAOp::Xvi4ger8:
    return "llvm.ppc.mma.xvi4ger8";
  case MMAOp::Xvi4ger8pp:
    return "llvm.ppc.mma.xvi4ger8pp";
  case MMAOp::Xvi8ger4:
    return "llvm.ppc.mma.xvi8ger4";
  case MMAOp::Xvi8ger4pp:
    return "llvm.ppc.mma.xvi8ger4pp";
  case MMAOp::Xvi8ger4spp:
    return "llvm.ppc.mma.xvi8ger4spp";
  }
  llvm_unreachable("getMmaIrIntrName");
}

mlir::FunctionType getMmaIrFuncType(mlir::MLIRContext *context, MMAOp mmaOp) {
  switch (mmaOp) {
  case MMAOp::AssembleAcc:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 4);
  case MMAOp::AssemblePair:
    return genMmaVpFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::DisassembleAcc:
    return genMmaDisassembleFuncType(context, mmaOp);
  case MMAOp::DisassemblePair:
    return genMmaDisassembleFuncType(context, mmaOp);
  case MMAOp::Xxmfacc:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 0);
  case MMAOp::Xxmtacc:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 0);
  case MMAOp::Xxsetaccz:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 0);
  case MMAOp::Pmxvbf16ger2:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvbf16ger2nn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvbf16ger2np:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvbf16ger2pn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvbf16ger2pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvf16ger2:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvf16ger2nn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvf16ger2np:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvf16ger2pn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvf16ger2pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvf32ger:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 2);
  case MMAOp::Pmxvf32gernn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 2);
  case MMAOp::Pmxvf32gernp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 2);
  case MMAOp::Pmxvf32gerpn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 2);
  case MMAOp::Pmxvf32gerpp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 2);
  case MMAOp::Pmxvf64ger:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 1, /*Vector*/ 1,
                            /*Integer*/ 2);
  case MMAOp::Pmxvf64gernn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 1, /*Vector*/ 1,
                            /*Integer*/ 2);
  case MMAOp::Pmxvf64gernp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 1, /*Vector*/ 1,
                            /*Integer*/ 2);
  case MMAOp::Pmxvf64gerpn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 1, /*Vector*/ 1,
                            /*Integer*/ 2);
  case MMAOp::Pmxvf64gerpp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 1, /*Vector*/ 1,
                            /*Integer*/ 2);
  case MMAOp::Pmxvi16ger2:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvi16ger2pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvi16ger2s:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvi16ger2spp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvi4ger8:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvi4ger8pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvi8ger4:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvi8ger4pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Pmxvi8ger4spp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2,
                            /*Integer*/ 3);
  case MMAOp::Xvbf16ger2:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvbf16ger2nn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvbf16ger2np:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvbf16ger2pn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvbf16ger2pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf16ger2:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf16ger2nn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf16ger2np:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf16ger2pn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf16ger2pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf32ger:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf32gernn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf32gernp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf32gerpn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf32gerpp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvf64ger:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 1, /*Vector*/ 1);
  case MMAOp::Xvf64gernn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 1, /*Vector*/ 1);
  case MMAOp::Xvf64gernp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 1, /*Vector*/ 1);
  case MMAOp::Xvf64gerpn:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 1, /*Vector*/ 1);
  case MMAOp::Xvf64gerpp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 1, /*Vector*/ 1);
  case MMAOp::Xvi16ger2:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvi16ger2pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvi16ger2s:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvi16ger2spp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvi4ger8:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvi4ger8pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvi8ger4:
    return genMmaVqFuncType(context, /*Quad*/ 0, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvi8ger4pp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  case MMAOp::Xvi8ger4spp:
    return genMmaVqFuncType(context, /*Quad*/ 1, /*Pair*/ 0, /*Vector*/ 2);
  }
  llvm_unreachable("getMmaIrFuncType");
}

template <MMAOp IntrId, MMAHandlerOp HandlerOp>
void PPCIntrinsicLibrary::genMmaIntr(llvm::ArrayRef<fir::ExtendedValue> args) {
  auto context{builder.getContext()};
  mlir::FunctionType intrFuncType{getMmaIrFuncType(context, IntrId)};
  mlir::func::FuncOp funcOp{
      builder.addNamedFunction(loc, getMmaIrIntrName(IntrId), intrFuncType)};
  llvm::SmallVector<mlir::Value> intrArgs;

  // Depending on SubToFunc, change the subroutine call to a function call.
  // First argument represents the result. Rest of the arguments
  // are shifted one position to form the actual argument list.
  size_t argStart{0};
  size_t argStep{1};
  size_t e{args.size()};
  if (HandlerOp == MMAHandlerOp::SubToFunc) {
    // The first argument becomes function result. Start from the second
    // argument.
    argStart = 1;
  } else if (HandlerOp == MMAHandlerOp::SubToFuncReverseArgOnLE) {
    // Reverse argument order on little-endian target only.
    // The reversal does not depend on the setting of non-native-order option.
    if (Fortran::evaluate::isHostLittleEndian) {
      // Load the arguments in reverse order.
      argStart = args.size() - 1;
      // The first argument becomes function result. Stop at the second
      // argument.
      e = 0;
      argStep = -1;
    } else {
      // Load the arguments in natural order.
      // The first argument becomes function result. Start from the second
      // argument.
      argStart = 1;
    }
  }

  for (size_t i = argStart, j = 0; i != e; i += argStep, ++j) {
    auto v{fir::getBase(args[i])};
    if (i == 0 && HandlerOp == MMAHandlerOp::FirstArgIsResult) {
      // First argument is passed in as an address. We need to load
      // the content to match the LLVM interface.
      v = builder.create<fir::LoadOp>(loc, v);
    }
    auto vType{v.getType()};
    mlir::Type targetType{intrFuncType.getInput(j)};
    if (vType != targetType) {
      if (targetType.isa<mlir::VectorType>()) {
        // Perform vector type conversion for arguments passed by value.
        auto eleTy{vType.dyn_cast<fir::VectorType>().getEleTy()};
        auto len{vType.dyn_cast<fir::VectorType>().getLen()};
        mlir::VectorType mlirType = mlir::VectorType::get(len, eleTy);
        auto v0{builder.createConvert(loc, mlirType, v)};
        auto v1{builder.create<mlir::vector::BitCastOp>(loc, targetType, v0)};
        intrArgs.push_back(v1);
      } else if (targetType.isa<mlir::IntegerType>() &&
                 vType.isa<mlir::IntegerType>()) {
        auto v0{builder.createConvert(loc, targetType, v)};
        intrArgs.push_back(v0);
      } else {
        llvm::errs() << "\nUnexpected type conversion requested: "
                     << " from " << vType << " to " << targetType << "\n";
        llvm_unreachable("Unsupported type conversion for argument to PowerPC "
                         "MMA intrinsic");
      }
    } else {
      intrArgs.push_back(v);
    }
  }
  auto callSt{builder.create<fir::CallOp>(loc, funcOp, intrArgs)};
  if (HandlerOp == MMAHandlerOp::SubToFunc ||
      HandlerOp == MMAHandlerOp::SubToFuncReverseArgOnLE ||
      HandlerOp == MMAHandlerOp::FirstArgIsResult) {
    // Convert pointer type if needed.
    mlir::Value callResult{callSt.getResult(0)};
    mlir::Value destPtr{fir::getBase(args[0])};
    mlir::Type callResultPtrType{builder.getRefType(callResult.getType())};
    if (destPtr.getType() != callResultPtrType) {
      destPtr = builder.create<fir::ConvertOp>(loc, callResultPtrType, destPtr);
    }
    // Copy the result.
    builder.create<fir::StoreOp>(loc, callResult, destPtr);
  }
}

} // namespace fir
