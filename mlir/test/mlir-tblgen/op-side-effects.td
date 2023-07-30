// RUN: mlir-tblgen -gen-op-defs -I %S/../../include %s | FileCheck %s

include "mlir/Interfaces/SideEffectInterfaces.td"

def TEST_Dialect : Dialect {
  let name = "test";
}
class TEST_Op<string mnemonic, list<Trait> traits = []> :
    Op<TEST_Dialect, mnemonic, traits>;

def CustomResource : Resource<"CustomResource">;

def SideEffectOpA : TEST_Op<"side_effect_op_a"> {
  let arguments = (ins
    Arg<Variadic<AnyMemRef>, "", [MemRead]>,
    Arg<AnyMemRef, "", [MemWriteAt<1>]>,
    Arg<SymbolRefAttr, "", [MemRead]>:$symbol,
    Arg<FlatSymbolRefAttr, "", [MemWrite]>:$flat_symbol,
    Arg<OptionalAttr<SymbolRefAttr>, "", [MemRead]>:$optional_symbol
  );
  let results = (outs Res<AnyMemRef, "", [MemAlloc<CustomResource, 0>]>);
}

def SideEffectOpB : TEST_Op<"side_effect_op_b",
    [MemoryEffects<[MemWrite<CustomResource, 0>]>]>;

// CHECK: void SideEffectOpA::getEffects
// CHECK:   for (::mlir::Value value : getODSOperands(0))
// CEHCK:     mlir::StringAttr attrName = mlir::StringAttr::get(getContext(), "SideEffectOrderAttr");
// CHECK:     mlir::IntegerAttr stageAttr = mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), 0);
// CHECK:     SmallVector<mlir::NamedAttribute, 1> attrs;
// CHECK:     attrs.push_back(mlir::NamedAttribute(attrName, stageAttr));
// CHECK:     mlir::DictionaryAttr paramAttr = mlir::DictionaryAttr::get(getContext(), attrs);
// CHECK:     effects.emplace_back(::mlir::MemoryEffects::Read::get(), value, paramAttr, ::mlir::SideEffects::DefaultResource::get());
// CHECK:   for (::mlir::Value value : getODSOperands(1))
// CEHCK:     mlir::StringAttr attrName = mlir::StringAttr::get(getContext(), "SideEffectOrderAttr");
// CHECK:     mlir::IntegerAttr stageAttr = mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), 1);
// CHECK:     SmallVector<mlir::NamedAttribute, 1> attrs;
// CHECK:     attrs.push_back(mlir::NamedAttribute(attrName, stageAttr));
// CHECK:     mlir::DictionaryAttr paramAttr = mlir::DictionaryAttr::get(getContext(), attrs);
// CHECK:     effects.emplace_back(::mlir::MemoryEffects::Write::get(), value, paramAttr, ::mlir::SideEffects::DefaultResource::get());
// CHECK:   effects.emplace_back(::mlir::MemoryEffects::Read::get(), getSymbolAttr(), paramAttr, ::mlir::SideEffects::DefaultResource::get());
// CHECK:   effects.emplace_back(::mlir::MemoryEffects::Write::get(), getFlatSymbolAttr(), paramAttr, ::mlir::SideEffects::DefaultResource::get());
// CHECK:   if (auto symbolRef = getOptionalSymbolAttr())
// CHECK:     effects.emplace_back(::mlir::MemoryEffects::Read::get(), symbolRef, paramAttr, ::mlir::SideEffects::DefaultResource::get());
// CHECK:   for (::mlir::Value value : getODSResults(0))
// CHECK:     effects.emplace_back(::mlir::MemoryEffects::Allocate::get(), value, paramAttr, CustomResource::get());

// CHECK: void SideEffectOpB::getEffects
// CEHCK:   mlir::StringAttr attrName = mlir::StringAttr::get(getContext(), "SideEffectOrderAttr");
// CHECK:   mlir::IntegerAttr stageAttr = mlir::IntegerAttr::get(mlir::IndexType::get(getContext()), 0);
// CHECK:   SmallVector<mlir::NamedAttribute, 1> attrs;
// CHECK:   attrs.push_back(mlir::NamedAttribute(attrName, stageAttr));
// CHECK:   mlir::DictionaryAttr paramAttr = mlir::DictionaryAttr::get(getContext(), attrs);
// CHECK:   effects.emplace_back(::mlir::MemoryEffects::Write::get(), paramAttr, CustomResource::get());
