#include "flang/Optimizer/Dialect/FIRDialect.h"
#include "flang/Optimizer/Dialect/FIROps.h"
#include "flang/Optimizer/Dialect/FIRType.h"
#include "flang/Optimizer/Support/InternalNames.h"
#include "flang/Optimizer/Transforms/Passes.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/OpenMP/OpenMPDialect.h"
#include "mlir/IR/BuiltinDialect.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/SymbolTable.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/LLVM.h"
#include "mlir/Transforms/RegionUtils.h"
#include "llvm/Frontend/OpenMP/OMPIRBuilder.h"

namespace fir {
#define GEN_PASS_DEF_OMPEARLYOUTLININGPASS
#include "flang/Optimizer/Transforms/Passes.h.inc"
} // namespace fir

namespace {
class OMPEarlyOutliningPass
    : public fir::impl::OMPEarlyOutliningPassBase<OMPEarlyOutliningPass> {

  std::string getOutlinedFnName(llvm::StringRef parentName, unsigned count) {
    return std::string(parentName) + "_omp_outline_" + std::to_string(count);
  }

  bool isDeclareTargetOp(mlir::Operation *op) {
    if (fir::AddrOfOp addressOfOp = mlir::dyn_cast<fir::AddrOfOp>(op))
      if (fir::GlobalOp gOp = mlir::dyn_cast<fir::GlobalOp>(
              addressOfOp->getParentOfType<mlir::ModuleOp>().lookupSymbol(
                  addressOfOp.getSymbol())))
        if (auto declareTargetGlobal =
                llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
                    gOp.getOperation()))
          if (declareTargetGlobal.isDeclareTarget())
            return true;
    return false;
  }

  mlir::func::FuncOp outlineTargetOp(mlir::OpBuilder &builder,
                                     mlir::omp::TargetOp &targetOp,
                                     mlir::func::FuncOp &parentFunc,
                                     unsigned count) {
    // Collect inputs
    llvm::SetVector<mlir::Value> inputs;
    for (auto operand : targetOp.getOperation()->getOperands())
      inputs.insert(operand);

    mlir::Region &targetRegion = targetOp.getRegion();
    mlir::getUsedValuesDefinedAbove(targetRegion, inputs);

    // filter out declareTarget entries which are specially handled
    // at the moment, we do not wish these to end up as function arguments
    // as they do not end up as kernel arguments.
    for (auto value : inputs)
      if (value.getDefiningOp())
        if (isDeclareTargetOp(value.getDefiningOp()))
          inputs.remove(value);

    // Create new function and initialize
    mlir::FunctionType funcType = builder.getFunctionType(
        mlir::TypeRange(inputs.getArrayRef()), mlir::TypeRange());
    std::string parentName(parentFunc.getName());
    std::string funcName = getOutlinedFnName(parentName, count);
    auto loc = targetOp.getLoc();
    mlir::func::FuncOp newFunc =
        mlir::func::FuncOp::create(loc, funcName, funcType);
    mlir::Block *entryBlock = newFunc.addEntryBlock();
    builder.setInsertionPointToStart(entryBlock);
    mlir::ValueRange newInputs = entryBlock->getArguments();

    // Set the declare target information, the outlined function
    // is always a host function.
    if (auto parentDTOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
            parentFunc.getOperation()))
      if (auto newDTOp = llvm::dyn_cast<mlir::omp::DeclareTargetInterface>(
              newFunc.getOperation()))
        newDTOp.setDeclareTarget(mlir::omp::DeclareTargetDeviceType::host,
                                 parentDTOp.getDeclareTargetCaptureClause());

    // Set the early outlining interface parent name
    if (auto earlyOutlineOp =
            llvm::dyn_cast<mlir::omp::EarlyOutliningInterface>(
                newFunc.getOperation()))
      earlyOutlineOp.setParentName(parentName);

    // Special handling for declare target mapped variables, declare
    // target has its addressOfOp cloned over, whereas we skip it for
    // regular map variables. This is because We need knowledge of
    // which global is linked to the map operation for declare
    // target, whereas we aren't bothered for the regular map
    // variables for the moment. We could treat both the same,
    // however, cloning across the minimum for the moment to avoid
    // optimisations breaking segments of the lowering seems prudent
    // as this was the original intent of the pass.
    mlir::IRMapping valueMap;
    for (auto oper : targetOp.getMapOperands()) {
      if (oper.getDefiningOp() && isDeclareTargetOp(oper.getDefiningOp())) {
        fir::AddrOfOp addrOp =
            mlir::dyn_cast<fir::AddrOfOp>(oper.getDefiningOp());
        mlir::Value newV = builder.clone(*addrOp)->getResult(0);
        valueMap.map(addrOp, newV);
      }
    }

    // Create input map from inputs to function parameters.
    for (auto InArg : llvm::zip(inputs, newInputs))
      valueMap.map(std::get<0>(InArg), std::get<1>(InArg));

    // Clone the target op into the new function
    builder.clone(*(targetOp.getOperation()), valueMap);

    // Create return op
    builder.create<mlir::func::ReturnOp>(loc);

    return newFunc;
  }

  // Returns true if a target region was found int the function.
  bool outlineTargetOps(mlir::OpBuilder &builder,
                        mlir::func::FuncOp &functionOp,
                        mlir::ModuleOp &moduleOp,
                        llvm::SmallVectorImpl<mlir::func::FuncOp> &newFuncs) {
    unsigned count = 0;
    for (auto TargetOp : functionOp.getOps<mlir::omp::TargetOp>()) {
      mlir::func::FuncOp outlinedFunc =
          outlineTargetOp(builder, TargetOp, functionOp, count);
      newFuncs.push_back(outlinedFunc);
      count++;
    }
    return count > 0;
  }

  void runOnOperation() override {
    mlir::ModuleOp moduleOp = getOperation();
    mlir::MLIRContext *context = &getContext();
    mlir::OpBuilder builder(context);
    llvm::SmallVector<mlir::func::FuncOp> newFuncs;

    for (auto functionOp :
         llvm::make_early_inc_range(moduleOp.getOps<mlir::func::FuncOp>())) {
      bool outlined = outlineTargetOps(builder, functionOp, moduleOp, newFuncs);
      if (outlined)
        functionOp.erase();
    }

    for (auto newFunc : newFuncs)
      moduleOp.push_back(newFunc);
  }
};

} // namespace

namespace fir {
std::unique_ptr<mlir::OperationPass<mlir::ModuleOp>>
createOMPEarlyOutliningPass() {
  return std::make_unique<OMPEarlyOutliningPass>();
}
} // namespace fir
