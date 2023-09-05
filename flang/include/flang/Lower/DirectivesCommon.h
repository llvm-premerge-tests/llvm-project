//===-- Lower/DirectivesCommon.h --------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Coding style: https://mlir.llvm.org/getting_started/DeveloperGuide/
//
//===----------------------------------------------------------------------===//
///
/// A location to place directive utilities shared across multiple lowering
/// files, e.g. utilities shared in OpenMP and OpenACC
//===----------------------------------------------------------------------===//

#ifndef FORTRAN_LOWER_DIRECTIVESCOMMON_H
#define FORTRAN_LOWER_DIRECTIVESCOMMON_H

#include "flang/Evaluate/tools.h"
#include "flang/Lower/AbstractConverter.h"
#include "flang/Lower/StatementContext.h"
#include "flang/Lower/Support/Utils.h"
#include "flang/Optimizer/Builder/BoxValue.h"
#include "flang/Optimizer/Builder/FIRBuilder.h"
#include "flang/Optimizer/Builder/Todo.h"
#include "flang/Semantics/tools.h"

#include "mlir/IR/Value.h"
#include <list>

namespace Fortran::lower {

using CreateDataBoundsOpFuncTy = std::function<mlir::Value(
    mlir::Location loc, mlir::Type boundTy, mlir::Value lb, mlir::Value ub,
    mlir::Value ext, mlir::Value stride, bool inBytes, mlir::Value base,
    fir::FirOpBuilder &builder)>;

// Default CreateDataBoundsOp that conforms to the CreateDataBoundsOpFuncTy
// interface
template <typename T>
static mlir::Value
createDataBoundsOp(mlir::Location loc, mlir::Type boundTy, mlir::Value lb,
                   mlir::Value ub, mlir::Value ext, mlir::Value stride,
                   bool inBytes, mlir::Value base, fir::FirOpBuilder &builder) {
  return builder.create<T>(loc, boundTy, lb, ub, ext, stride, inBytes, base);
}

mlir::Value getDataOperandBaseAddr(Fortran::lower::AbstractConverter &converter,
                                   fir::FirOpBuilder &builder,
                                   Fortran::lower::SymbolRef sym,
                                   mlir::Location loc);

/// Generate the bounds operation from the descriptor information.
llvm::SmallVector<mlir::Value>
genBoundsOpsFromBox(fir::FirOpBuilder &builder, mlir::Location loc,
                    Fortran::lower::AbstractConverter &converter,
                    fir::ExtendedValue dataExv, mlir::Value box,
                    mlir::Type boundTy,
                    CreateDataBoundsOpFuncTy createDataBound);

/// Generate bounds operation for base array without any subscripts
/// provided.
llvm::SmallVector<mlir::Value>
genBaseBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
                 Fortran::lower::AbstractConverter &converter,
                 fir::ExtendedValue dataExv, mlir::Value baseAddr,
                 mlir::Type boundTy, CreateDataBoundsOpFuncTy createDataBound);

/// Generate bounds operations for an array section when subscripts are
/// provided.
llvm::SmallVector<mlir::Value>
genBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
             Fortran::lower::AbstractConverter &converter,
             Fortran::lower::StatementContext &stmtCtx,
             const std::list<Fortran::parser::SectionSubscript> &subscripts,
             std::stringstream &asFortran, fir::ExtendedValue &dataExv,
             mlir::Value baseAddr, mlir::Type boundTy,
             CreateDataBoundsOpFuncTy createDataBound);

mlir::Value gatherDataOperandAddrAndBounds(
    Fortran::lower::AbstractConverter &converter, fir::FirOpBuilder &builder,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    const std::variant<Fortran::parser::Designator, Fortran::parser::Name>
        &objectName,
    mlir::Location operandLocation, std::stringstream &asFortran,
    mlir::Type boundTy, CreateDataBoundsOpFuncTy createDataBound,
    llvm::SmallVector<mlir::Value> &bounds);

} // namespace Fortran::lower

#endif // FORTRAN_LOWER_DIRECTIVESCOMMON_H
