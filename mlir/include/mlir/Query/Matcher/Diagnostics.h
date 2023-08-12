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

namespace mlir::query::matcher::internal {

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
  // All errors from the system.
  enum class ErrorType {
    None,

    // Parser Errors
    ParserFailedToBuildMatcher,
    ParserInvalidToken,
    ParserNoCloseParen,
    ParserNoCode,
    ParserNoComma,
    ParserNoOpenParen,
    ParserNotAMatcher,
    ParserOverloadedType,
    ParserStringError,
    ParserTrailingCode,

    // Registry Errors
    RegistryMatcherNotFound,
    RegistryValueNotFound,
    RegistryWrongArgCount,
    RegistryWrongArgType
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

  // Add an error message with the specified range and error type.
  // Returns an ArgStream object to allow constructing the error message using
  // the << operator.
  ArgStream addError(SourceRange range, ErrorType error);

  // Print all error messages to the specified output stream.
  void print(llvm::raw_ostream &os) const;

private:
  // Information stored for one frame of the context.
  struct ContextFrame {
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

  void printMessage(const ErrorContent::Message &message,
                    const llvm::Twine Prefix, llvm::raw_ostream &os) const;

  void printErrorContent(const ErrorContent &content,
                         llvm::raw_ostream &os) const;

  std::vector<ContextFrame> contextStack;
  std::vector<ErrorContent> errorValues;
};

} // namespace mlir::query::matcher::internal

#endif // MLIR_TOOLS_MLIRQUERY_MATCHER_DIAGNOSTICS_H
