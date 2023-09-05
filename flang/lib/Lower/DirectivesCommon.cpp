//===-- DirectivesCommon.cpp -------------------------------------------===//
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

#include "flang/Lower/DirectivesCommon.h"

namespace Fortran::lower {

mlir::Value getDataOperandBaseAddr(Fortran::lower::AbstractConverter &converter,
                                   fir::FirOpBuilder &builder,
                                   Fortran::lower::SymbolRef sym,
                                   mlir::Location loc) {
  mlir::Value symAddr = converter.getSymbolAddress(sym);
  // TODO: Might need revisiting to handle for non-shared clauses
  if (!symAddr) {
    if (const auto *details =
            sym->detailsIf<Fortran::semantics::HostAssocDetails>())
      symAddr = converter.getSymbolAddress(details->symbol());
  }

  if (!symAddr)
    llvm::report_fatal_error("could not retrieve symbol address");

  if (auto boxTy =
          fir::unwrapRefType(symAddr.getType()).dyn_cast<fir::BaseBoxType>()) {
    if (boxTy.getEleTy().isa<fir::RecordType>())
      TODO(loc, "derived type");

    // Load the box when baseAddr is a `fir.ref<fir.box<T>>` or a
    // `fir.ref<fir.class<T>>` type.
    if (symAddr.getType().isa<fir::ReferenceType>())
      return builder.create<fir::LoadOp>(loc, symAddr);
  }
  return symAddr;
}

/// Generate the bounds operation from the descriptor information.
llvm::SmallVector<mlir::Value>
genBoundsOpsFromBox(fir::FirOpBuilder &builder, mlir::Location loc,
                    Fortran::lower::AbstractConverter &converter,
                    fir::ExtendedValue dataExv, mlir::Value box,
                    mlir::Type boundTy,
                    CreateDataBoundsOpFuncTy createDataBound) {
  llvm::SmallVector<mlir::Value> bounds;
  mlir::Type idxTy = builder.getIndexType();
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  assert(box.getType().isa<fir::BaseBoxType>() &&
         "expect fir.box or fir.class");
  for (unsigned dim = 0; dim < dataExv.rank(); ++dim) {
    mlir::Value d = builder.createIntegerConstant(loc, idxTy, dim);
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    auto dimInfo =
        builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy, box, d);
    mlir::Value lb = builder.createIntegerConstant(loc, idxTy, 0);
    mlir::Value ub =
        builder.create<mlir::arith::SubIOp>(loc, dimInfo.getExtent(), one);
    mlir::Value bound =
        createDataBound(loc, boundTy, lb, ub, mlir::Value(),
                        dimInfo.getByteStride(), true, baseLb, builder);
    bounds.push_back(bound);
  }
  return bounds;
}

/// Generate bounds operation for base array without any subscripts
/// provided.
llvm::SmallVector<mlir::Value>
genBaseBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
                 Fortran::lower::AbstractConverter &converter,
                 fir::ExtendedValue dataExv, mlir::Value baseAddr,
                 mlir::Type boundTy, CreateDataBoundsOpFuncTy createDataBound) {
  mlir::Type idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> bounds;

  if (dataExv.rank() == 0)
    return bounds;

  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  for (std::size_t dim = 0; dim < dataExv.rank(); ++dim) {
    mlir::Value baseLb =
        fir::factory::readLowerBound(builder, loc, dataExv, dim, one);
    mlir::Value ext = fir::factory::readExtent(builder, loc, dataExv, dim);
    mlir::Value lb = builder.createIntegerConstant(loc, idxTy, 0);

    // ub = extent - 1
    mlir::Value ub = builder.create<mlir::arith::SubIOp>(loc, ext, one);
    mlir::Value bound =
        createDataBound(loc, boundTy, lb, ub, ext, one, false, baseLb, builder);
    bounds.push_back(bound);
  }
  return bounds;
}

/// Generate bounds operations for an array section when subscripts are
/// provided.
llvm::SmallVector<mlir::Value>
genBoundsOps(fir::FirOpBuilder &builder, mlir::Location loc,
             Fortran::lower::AbstractConverter &converter,
             Fortran::lower::StatementContext &stmtCtx,
             const std::list<Fortran::parser::SectionSubscript> &subscripts,
             std::stringstream &asFortran, fir::ExtendedValue &dataExv,
             mlir::Value baseAddr, mlir::Type boundTy,
             CreateDataBoundsOpFuncTy createDataBound) {
  int dimension = 0;
  mlir::Type idxTy = builder.getIndexType();
  llvm::SmallVector<mlir::Value> bounds;

  mlir::Value zero = builder.createIntegerConstant(loc, idxTy, 0);
  mlir::Value one = builder.createIntegerConstant(loc, idxTy, 1);
  for (const auto &subscript : subscripts) {
    if (const auto *triplet{
            std::get_if<Fortran::parser::SubscriptTriplet>(&subscript.u)}) {
      if (dimension != 0)
        asFortran << ',';
      mlir::Value lbound, ubound, extent;
      std::optional<std::int64_t> lval, uval;
      mlir::Value baseLb =
          fir::factory::readLowerBound(builder, loc, dataExv, dimension, one);
      bool defaultLb = baseLb == one;
      mlir::Value stride = one;
      bool strideInBytes = false;

      if (fir::unwrapRefType(baseAddr.getType()).isa<fir::BaseBoxType>()) {
        mlir::Value d = builder.createIntegerConstant(loc, idxTy, dimension);
        auto dimInfo = builder.create<fir::BoxDimsOp>(loc, idxTy, idxTy, idxTy,
                                                      baseAddr, d);
        stride = dimInfo.getByteStride();
        strideInBytes = true;
      }

      const auto &lower{std::get<0>(triplet->t)};
      if (lower) {
        lval = Fortran::semantics::GetIntValue(lower);
        if (lval) {
          if (defaultLb) {
            lbound = builder.createIntegerConstant(loc, idxTy, *lval - 1);
          } else {
            mlir::Value lb = builder.createIntegerConstant(loc, idxTy, *lval);
            lbound = builder.create<mlir::arith::SubIOp>(loc, lb, baseLb);
          }
          asFortran << *lval;
        } else {
          const Fortran::lower::SomeExpr *lexpr =
              Fortran::semantics::GetExpr(*lower);
          mlir::Value lb =
              fir::getBase(converter.genExprValue(loc, *lexpr, stmtCtx));
          lb = builder.createConvert(loc, baseLb.getType(), lb);
          lbound = builder.create<mlir::arith::SubIOp>(loc, lb, baseLb);
          asFortran << lexpr->AsFortran();
        }
      } else {
        lbound = defaultLb ? zero : baseLb;
      }
      asFortran << ':';
      const auto &upper{std::get<1>(triplet->t)};
      if (upper) {
        uval = Fortran::semantics::GetIntValue(upper);
        if (uval) {
          if (defaultLb) {
            ubound = builder.createIntegerConstant(loc, idxTy, *uval - 1);
          } else {
            mlir::Value ub = builder.createIntegerConstant(loc, idxTy, *uval);
            ubound = builder.create<mlir::arith::SubIOp>(loc, ub, baseLb);
          }
          asFortran << *uval;
        } else {
          const Fortran::lower::SomeExpr *uexpr =
              Fortran::semantics::GetExpr(*upper);
          mlir::Value ub =
              fir::getBase(converter.genExprValue(loc, *uexpr, stmtCtx));
          ub = builder.createConvert(loc, baseLb.getType(), ub);
          ubound = builder.create<mlir::arith::SubIOp>(loc, ub, baseLb);
          asFortran << uexpr->AsFortran();
        }
      }
      if (lower && upper) {
        if (lval && uval && *uval < *lval) {
          mlir::emitError(loc, "zero sized array section");
          break;
        } else if (std::get<2>(triplet->t)) {
          const auto &strideExpr{std::get<2>(triplet->t)};
          if (strideExpr) {
            mlir::emitError(loc, "stride cannot be specified on "
                                 "an OpenMP array section");
            break;
          }
        }
      }
      // ub = baseLb + extent - 1
      if (!ubound) {
        mlir::Value ext =
            fir::factory::readExtent(builder, loc, dataExv, dimension);
        mlir::Value lbExt =
            builder.create<mlir::arith::AddIOp>(loc, ext, baseLb);
        ubound = builder.create<mlir::arith::SubIOp>(loc, lbExt, one);
      }
      mlir::Value bound =
          createDataBound(loc, boundTy, lbound, ubound, extent, stride,
                          strideInBytes, baseLb, builder);
      bounds.push_back(bound);
      ++dimension;
    }
  }
  return bounds;
}

mlir::Value gatherDataOperandAddrAndBounds(
    Fortran::lower::AbstractConverter &converter, fir::FirOpBuilder &builder,
    Fortran::semantics::SemanticsContext &semanticsContext,
    Fortran::lower::StatementContext &stmtCtx,
    const std::variant<Fortran::parser::Designator, Fortran::parser::Name>
        &objectName,
    mlir::Location operandLocation, std::stringstream &asFortran,
    mlir::Type boundTy, CreateDataBoundsOpFuncTy createDataBound,
    llvm::SmallVector<mlir::Value> &bounds) {
  mlir::Value baseAddr;

  std::visit(
      Fortran::common::visitors{
          [&](const Fortran::parser::Designator &designator) {
            if (auto expr{Fortran::semantics::AnalyzeExpr(semanticsContext,
                                                          designator)}) {
              if ((*expr).Rank() > 0 &&
                  Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                      designator)) {
                const auto *arrayElement =
                    Fortran::parser::Unwrap<Fortran::parser::ArrayElement>(
                        designator);
                const auto *dataRef =
                    std::get_if<Fortran::parser::DataRef>(&designator.u);
                fir::ExtendedValue dataExv;
                if (Fortran::parser::Unwrap<
                        Fortran::parser::StructureComponent>(
                        arrayElement->base)) {
                  auto exprBase = Fortran::semantics::AnalyzeExpr(
                      semanticsContext, arrayElement->base);
                  dataExv = converter.genExprAddr(operandLocation, *exprBase,
                                                  stmtCtx);
                  baseAddr = fir::getBase(dataExv);
                  asFortran << (*exprBase).AsFortran();
                } else {
                  const Fortran::parser::Name &name =
                      Fortran::parser::GetLastName(*dataRef);
                  baseAddr = getDataOperandBaseAddr(
                      converter, builder, *name.symbol, operandLocation);
                  dataExv = converter.getSymbolExtendedValue(*name.symbol);
                  asFortran << name.ToString();
                }

                if (!arrayElement->subscripts.empty()) {
                  asFortran << '(';
                  bounds =
                      genBoundsOps(builder, operandLocation, converter, stmtCtx,
                                   arrayElement->subscripts, asFortran, dataExv,
                                   baseAddr, boundTy, createDataBound);
                }
                asFortran << ')';
              } else if (Fortran::parser::Unwrap<
                             Fortran::parser::StructureComponent>(designator)) {
                fir::ExtendedValue compExv =
                    converter.genExprAddr(operandLocation, *expr, stmtCtx);
                baseAddr = fir::getBase(compExv);
                if (fir::unwrapRefType(baseAddr.getType())
                        .isa<fir::SequenceType>())
                  bounds = genBaseBoundsOps(builder, operandLocation, converter,
                                            compExv, baseAddr, boundTy,
                                            createDataBound);
                asFortran << (*expr).AsFortran();

                // If the component is an allocatable or pointer the result of
                // genExprAddr will be the result of a fir.box_addr operation.
                // Retrieve the box so we handle it like other descriptor.
                if (auto boxAddrOp = mlir::dyn_cast_or_null<fir::BoxAddrOp>(
                        baseAddr.getDefiningOp())) {
                  baseAddr = boxAddrOp.getVal();
                  bounds = genBoundsOpsFromBox(builder, operandLocation,
                                               converter, compExv, baseAddr,
                                               boundTy, createDataBound);
                }
              } else {
                // Scalar or full array.
                if (const auto *dataRef{
                        std::get_if<Fortran::parser::DataRef>(&designator.u)}) {
                  const Fortran::parser::Name &name =
                      Fortran::parser::GetLastName(*dataRef);
                  fir::ExtendedValue dataExv =
                      converter.getSymbolExtendedValue(*name.symbol);
                  baseAddr = getDataOperandBaseAddr(
                      converter, builder, *name.symbol, operandLocation);
                  if (fir::unwrapRefType(baseAddr.getType())
                          .isa<fir::BaseBoxType>())
                    bounds = genBoundsOpsFromBox(builder, operandLocation,
                                                 converter, dataExv, baseAddr,
                                                 boundTy, createDataBound);
                  if (fir::unwrapRefType(baseAddr.getType())
                          .isa<fir::SequenceType>())
                    bounds = genBaseBoundsOps(builder, operandLocation,
                                              converter, dataExv, baseAddr,
                                              boundTy, createDataBound);
                  asFortran << name.ToString();
                } else { // Unsupported
                  llvm::report_fatal_error(
                      "Unsupported type of OpenACC operand");
                }
              }
            }
          },
          [&](const Fortran::parser::Name &name) {
            baseAddr = getDataOperandBaseAddr(converter, builder, *name.symbol,
                                              operandLocation);
            asFortran << name.ToString();
          }},
      objectName);
  return baseAddr;
}

} // namespace Fortran::lower
