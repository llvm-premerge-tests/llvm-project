//===--- Diagnostics.h - Helper class for error diagnostics -----*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// Diagnostics class to manage error messages. Implementation shares similarity
// to clang-query Diagnostics.
//
//===----------------------------------------------------------------------===//

#ifndef MLIR_TOOLS_MLIRQUERY_MATCHER_DIAGNOSTICS_H
#define MLIR_TOOLS_MLIRQUERY_MATCHER_DIAGNOSTICS_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/raw_ostream.h"
#include <string>
#include <vector>

namespace mlir::query::matcher {

// Represents the line and column numbers in a source query.
struct SourceLocation {
  unsigned line{};
  unsigned column{};
};

// Represents a range in a source query, defined by its start and end locations.
struct SourceRange {
  SourceLocation start{};
  SourceLocation end{};
};

// Diagnostics class to manage error messages.
class Diagnostics {
public:
  // Parser context types.
  enum ContextType { CT_MatcherArg, CT_MatcherConstruct };

  // All errors from the system.
  enum ErrorType {
    ET_None,

    // Parser Errors
    ET_ParserFailedToBuildMatcher,
    ET_ParserInvalidToken,
    ET_ParserNoCloseParen,
    ET_ParserNoCode,
    ET_ParserNoComma,
    ET_ParserNoOpenParen,
    ET_ParserNotAMatcher,
    ET_ParserOverloadedType,
    ET_ParserStringError,
    ET_ParserTrailingCode,

    // Registry Errors
    ET_RegistryMatcherNotFound,
    ET_RegistryValueNotFound,
    ET_RegistryWrongArgCount,
    ET_RegistryWrongArgType
  };

  // Helper stream class for constructing error messages.
  class ArgStream {
  public:
    ArgStream(std::vector<std::string> *out) : out(out) {}
    template <class T>
    ArgStream &operator<<(const T &arg) {
      return operator<<(llvm::Twine(arg));
    }
    ArgStream &operator<<(const llvm::Twine &arg);

  private:
    std::vector<std::string> *out;
  };

  // Context for constructing a matcher or parsing its argument.
  struct Context {
    enum ConstructMatcherEnum { ConstructMatcher };
    Context(ConstructMatcherEnum, Diagnostics *error,
            llvm::StringRef matcherName, SourceRange matcherRange);
    enum MatcherArgEnum { MatcherArg };
    Context(MatcherArgEnum, Diagnostics *error, llvm::StringRef matcherName,
            SourceRange matcherRange, int argNumber);
    ~Context();

  private:
    Diagnostics *const error;
  };

  // Context for managing overloaded matcher construction.
  struct OverloadContext {
    // Construct an overload context with the given error.
    OverloadContext(Diagnostics *error);
    ~OverloadContext();
    // Revert all errors that occurred within this context.
    void revertErrors();

  private:
    Diagnostics *const error;
    unsigned beginIndex{};
  };

  // Add an error message with the specified range and error type.
  // Returns an ArgStream object to allow constructing the error message using
  // the << operator.
  ArgStream addError(SourceRange range, ErrorType error);

  // Information stored for one frame of the context.
  struct ContextFrame {
    ContextType type;
    SourceRange range;
    std::vector<std::string> args;
  };

  // Information stored for each error found.
  struct ErrorContent {
    std::vector<ContextFrame> contextStack;
    struct Message {
      SourceRange range;
      ErrorType type;
      std::vector<std::string> args;
    };
    std::vector<Message> messages;
  };

  // Get an array reference to the error contents.
  llvm::ArrayRef<ErrorContent> errors() const { return errorValues; }

  // Print all error messages to the specified output stream.
  void print(llvm::raw_ostream &OS) const;

  // Print the full error messages, including the context information, to the
  // specified output stream.
  void printFull(llvm::raw_ostream &OS) const;

private:
  // Push a new context frame onto the context stack with the specified type and
  // range.
  ArgStream pushContextFrame(ContextType type, SourceRange range);

  std::vector<ContextFrame> contextStack;
  std::vector<ErrorContent> errorValues;
};

} // namespace mlir::query::matcher

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_DIAGNOSTICS_H
