//===- ExtractAPI/Serialization/SerializerBase.h ----------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
///
/// \file
/// This file defines the ExtractAPI APISetVisitor interface.
///
//===----------------------------------------------------------------------===//

#ifndef LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H
#define LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H

#include "clang/ExtractAPI/API.h"

namespace clang {
namespace extractapi {

/// The base interface of visitors for API information.
template <typename Derived> class APISetVisitor {
public:
  void traverseAPISet() {
    getDerived()->traverseGlobalVariableRecords();

    getDerived()->traverseGlobalFunctionRecords();

    getDerived()->traverseEnumRecords();

    getDerived()->traverseStaticFieldRecords();

    getDerived()->traverseCXXClassRecords();

    getDerived()->traverseClassTemplateRecords();

    getDerived()->traverseClassTemplateSpecRecords();

    getDerived()->traverseClassTemplatePartialSpecRecords();

    getDerived()->traverseConcepts();

    getDerived()->traverseGlobalVariableTemplateRecords();

    getDerived()->traverseGlobalVariableTemplateSpecRecords();

    getDerived()->traverseGlobalVariableTemplatePartialSpecRecords();

    getDerived()->traverseStructRecords();

    getDerived()->traverseObjCInterfaces();

    getDerived()->traverseObjCProtocols();

    getDerived()->traverseObjCCategories();

    getDerived()->traverseMacroDefinitionRecords();

    getDerived()->traverseTypedefRecords();
  }

  void traverseGlobalFunctionRecords() {
    for (const auto &GlobalFunction : API.getGlobalFunctions())
      getDerived()->visitGlobalFunctionRecord(*GlobalFunction.second);
  }

  void traverseGlobalVariableRecords() {
    for (const auto &GlobalVariable : API.getGlobalVariables())
      getDerived()->visitGlobalVariableRecord(*GlobalVariable.second);
  }

  void traverseEnumRecords() {
    for (const auto &Enum : API.getEnums())
      getDerived()->visitEnumRecord(*Enum.second);
  }

  void traverseStructRecords() {
    for (const auto &Struct : API.getStructs())
      getDerived()->visitStructRecord(*Struct.second);
  }

  void traverseStaticFieldRecords() {
    for (const auto &StaticField : API.getStaticFields())
      getDerived()->visitStaticFieldRecord(*StaticField.second);
  }

  void traverseCXXClassRecords() {
    for (const auto &Class : API.getCXXClasses())
      getDerived()->visitCXXClassRecord(*Class.second);
  }

  void traverseClassTemplateRecords() {
    for (const auto &ClassTemplate : API.getClassTemplates())
      getDerived()->visitClassTemplateRecord(*ClassTemplate.second);
  }

  void traverseClassTemplateSpecRecords() {
    for (const auto &ClassTemplateSpec : API.getClassTemplateSpecializations())
      getDerived()->visitClassTemplateSpecRecord(*ClassTemplateSpec.second);
  }

  void traverseClassTemplatePartialSpecRecords() {
    for (const auto &ClassTemplatePartialSpec :
         API.getClassTemplatePartialSpecializations())
      getDerived()->visitClassTemplatePartialSpecRecord(
          *ClassTemplatePartialSpec.second);
  }

  void traverseGlobalVariableTemplateRecords() {
    for (const auto &GlobalVariableTemplate : API.getGlobalVariableTemplates())
      getDerived()->visitGlobalVariableTemplateRecord(
          *GlobalVariableTemplate.second);
  }

  void traverseGlobalVariableTemplateSpecRecords() {
    for (const auto &GlobalVariableTemplateSpec :
         API.getGlobalVariableTemplateSpecializations())
      getDerived()->visitGlobalVariableTemplateSpecRecord(
          *GlobalVariableTemplateSpec.second);
  }

  void traverseGlobalVariableTemplatePartialSpecRecords() {
    for (const auto &GlobalVariableTemplatePartialSpec :
         API.getGlobalVariableTemplatePartialSpecializations())
      getDerived()->visitGlobalVariableTemplatePartialSpecRecord(
          *GlobalVariableTemplatePartialSpec.second);
  }

  void traverseConcepts() {
    for (const auto &Concept : API.getConcepts())
      getDerived()->visitConceptRecord(*Concept.second);
  }

  void traverseObjCInterfaces() {
    for (const auto &Interface : API.getObjCInterfaces())
      getDerived()->visitObjCContainerRecord(*Interface.second);
  }

  void traverseObjCProtocols() {
    for (const auto &Protocol : API.getObjCProtocols())
      getDerived()->visitObjCContainerRecord(*Protocol.second);
  }

  void traverseObjCCategories() {
    for (const auto &Category : API.getObjCCategories())
      getDerived()->visitObjCCategoryRecord(*Category.second);
  }

  void traverseMacroDefinitionRecords() {
    for (const auto &Macro : API.getMacros())
      getDerived()->visitMacroDefinitionRecord(*Macro.second);
  }

  void traverseTypedefRecords() {
    for (const auto &Typedef : API.getTypedefs())
      getDerived()->visitTypedefRecord(*Typedef.second);
  }

  /// Visit a global function record.
  void visitGlobalFunctionRecord(const GlobalFunctionRecord &Record){};

  /// Visit a global variable record.
  void visitGlobalVariableRecord(const GlobalVariableRecord &Record){};

  /// Visit an enum record.
  void visitEnumRecord(const EnumRecord &Record){};

  /// Visit a struct record.
  void visitStructRecord(const StructRecord &Record){};

  void visitStaticFieldRecord(const StaticFieldRecord &Record){};

  void visitCXXClassRecord(const CXXClassRecord &Record){};

  void visitClassTemplateRecord(const ClassTemplateRecord &Record){};

  void visitClassTemplateSpecRecord(const ClassTemplateSpecRecord &Record){};

  void visitClassTemplatePartialSpecRecord(
      const ClassTemplatePartialSpecRecord &Record){};

  void visitGlobalVariableTemplateRecord(
      const GlobalVariableTemplateRecord &Record) {}

  void visitGlobalVariableTemplateSpecRecord(
      const GlobalVariableTemplateSpecRecord &Record){};

  void visitGlobalVariableTemplatePartialSpecRecord(
      const GlobalVariableTemplatePartialSpecRecord &Record){};

  /// Visit an Objective-C container record.
  void visitObjCContainerRecord(const ObjCContainerRecord &Record){};

  /// Visit an Objective-C category record.
  void visitObjCCategoryRecord(const ObjCCategoryRecord &Record){};

  /// Visit a macro definition record.
  void visitMacroDefinitionRecord(const MacroDefinitionRecord &Record){};

  /// Visit a typedef record.
  void visitTypedefRecord(const TypedefRecord &Record){};

protected:
  const APISet &API;

public:
  APISetVisitor() = delete;
  APISetVisitor(const APISetVisitor &) = delete;
  APISetVisitor(APISetVisitor &&) = delete;
  APISetVisitor &operator=(const APISetVisitor &) = delete;
  APISetVisitor &operator=(APISetVisitor &&) = delete;

protected:
  APISetVisitor(const APISet &API) : API(API) {}
  ~APISetVisitor() = default;

  Derived *getDerived() { return static_cast<Derived *>(this); };
};

} // namespace extractapi
} // namespace clang

#endif // LLVM_CLANG_EXTRACTAPI_SERIALIZATION_SERIALIZERBASE_H
