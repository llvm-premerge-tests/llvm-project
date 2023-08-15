; RUN: %if spirv-registered-target %{ llc -O0 -mtriple=spirv64-unknown-unknown %s -o - | FileCheck %s --check-prefix=SPIRV-CHECK %}
; RUN: %if directx-registered-target %{ llc -O0 -mtriple=dxil-pc-shadermodel6.0-library %s -o - | FileCheck %s --check-prefix=DXIL-CHECK %}

; SPIRV-CHECK-DAG: OpCapability Addresses
; SPIRV-CHECK-DAG: OpCapability Linkage
; SPIRV-CHECK-DAG: OpCapability Kernel
; SPIRV-CHECK:     %1 = OpExtInstImport "OpenCL.std"
; SPIRV-CHECK:     OpMemoryModel Physical64 OpenCL
; SPIRV-CHECK:     OpSource Unknown 0

; DXIL-CHECK: target datalayout = "e-m:e-p:32:32-i1:32-i8:8-i16:16-i32:32-i64:64-f16:16-f32:32-f64:64-n8:16:32:64"
; DXIL-CHECK: target triple = "dxil-pc-shadermodel6.0-library"
