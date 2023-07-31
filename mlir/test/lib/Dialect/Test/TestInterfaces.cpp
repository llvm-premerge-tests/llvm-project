#include "TestInterfaces.h"

using namespace mlir;

bool mlir::TestEffects::Effect::classof(
    const mlir::SideEffects::Effect *effect) {
  return isa<mlir::TestEffects::Concrete>(effect);
}

LogicalResult mlir::detail::verifyTestExternalOpInterface(Operation *op) {
  if (op->getName().getStringRef().ends_with(
          "trigger_interface_verification_failure")) {
    return op->emitError() << "interface verification failure";
  }
  return success();
}
