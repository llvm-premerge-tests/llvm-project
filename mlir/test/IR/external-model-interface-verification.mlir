// RUN: mlir-opt %s --split-input-file --verify-diagnostics

test.succeed_interface_verification

// -----

// expected-error @below {{interface verification failure}}
test.trigger_interface_verification_failure
