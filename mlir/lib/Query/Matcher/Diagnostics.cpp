//===- MatcherDiagnostic.cpp ------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Query/Matcher/Diagnostics.h"

namespace mlir::query::matcher {

Diagnostics::ArgStream Diagnostics::pushContextFrame(ContextType type,
                                                     SourceRange range) {
  contextStack.emplace_back();
  ContextFrame &data = contextStack.back();
  data.type = type;
  data.range = range;
  return ArgStream(&data.args);
}

Diagnostics::Context::Context(ConstructMatcherEnum, Diagnostics *error,
                              llvm::StringRef matcherName,
                              SourceRange matcherRange)
    : error(error) {
  error->pushContextFrame(CT_MatcherConstruct, matcherRange) << matcherName;
}

Diagnostics::Context::Context(MatcherArgEnum, Diagnostics *error,
                              llvm::StringRef matcherName,
                              SourceRange matcherRange, int argnumber)
    : error(error) {
  error->pushContextFrame(CT_MatcherArg, matcherRange)
      << argnumber << matcherName;
}

Diagnostics::Context::~Context() { error->contextStack.pop_back(); }

Diagnostics::OverloadContext::OverloadContext(Diagnostics *error)
    : error(error), beginIndex(error->errorValues.size()) {}

Diagnostics::OverloadContext::~OverloadContext() {
  // Merge all errors that happened while in this context.
  if (beginIndex < error->errorValues.size()) {
    Diagnostics::ErrorContent &dest = error->errorValues[beginIndex];
    for (size_t i = beginIndex + 1, e = error->errorValues.size(); i < e; ++i) {
      dest.messages.push_back(error->errorValues[i].messages[0]);
    }
    error->errorValues.resize(beginIndex + 1);
  }
}

void Diagnostics::OverloadContext::revertErrors() {
  // Revert the errors.
  error->errorValues.resize(beginIndex);
}

Diagnostics::ArgStream &
Diagnostics::ArgStream::operator<<(const llvm::Twine &arg) {
  out->push_back(arg.str());
  return *this;
}

Diagnostics::ArgStream Diagnostics::addError(SourceRange range,
                                             ErrorType error) {
  errorValues.emplace_back();
  ErrorContent &last = errorValues.back();
  last.contextStack = contextStack;
  last.messages.emplace_back();
  last.messages.back().range = range;
  last.messages.back().type = error;
  return ArgStream(&last.messages.back().args);
}

static llvm::StringRef
contextTypeToFormatString(Diagnostics::ContextType type) {
  switch (type) {
  case Diagnostics::CT_MatcherConstruct:
    return "Error building matcher $0.";
  case Diagnostics::CT_MatcherArg:
    return "Error parsing argument $0 for matcher $1.";
  }
  llvm_unreachable("Unknown ContextType value.");
}

static llvm::StringRef errorTypeToFormatString(Diagnostics::ErrorType type) {
  switch (type) {
  case Diagnostics::ET_RegistryMatcherNotFound:
    return "Matcher not found: $0";
  case Diagnostics::ET_RegistryWrongArgCount:
    return "Incorrect argument count. (Expected = $0) != (Actual = $1)";
  case Diagnostics::ET_RegistryWrongArgType:
    return "Incorrect type for arg $0. (Expected = $1) != (Actual = $2)";
  case Diagnostics::ET_RegistryValueNotFound:
    return "Value not found: $0";

  case Diagnostics::ET_ParserStringError:
    return "Error parsing string token: <$0>";
  case Diagnostics::ET_ParserNoOpenParen:
    return "Error parsing matcher. Found token <$0> while looking for '('.";
  case Diagnostics::ET_ParserNoCloseParen:
    return "Error parsing matcher. Found end-of-code while looking for ')'.";
  case Diagnostics::ET_ParserNoComma:
    return "Error parsing matcher. Found token <$0> while looking for ','.";
  case Diagnostics::ET_ParserNoCode:
    return "End of code found while looking for token.";
  case Diagnostics::ET_ParserNotAMatcher:
    return "Input value is not a matcher expression.";
  case Diagnostics::ET_ParserInvalidToken:
    return "Invalid token <$0> found when looking for a value.";
  case Diagnostics::ET_ParserTrailingCode:
    return "Unexpected end of code.";
  case Diagnostics::ET_ParserOverloadedType:
    return "Input value has unresolved overloaded type: $0";
  case Diagnostics::ET_ParserFailedToBuildMatcher:
    return "Failed to build matcher: $0.";

  case Diagnostics::ET_None:
    return "<N/A>";
  }
  llvm_unreachable("Unknown ErrorType value.");
}

static void formatErrorString(llvm::StringRef formatString,
                              llvm::ArrayRef<std::string> args,
                              llvm::raw_ostream &OS) {
  while (!formatString.empty()) {
    std::pair<llvm::StringRef, llvm::StringRef> pieces =
        formatString.split("$");
    OS << pieces.first.str();
    if (pieces.second.empty())
      break;

    const char next = pieces.second.front();
    formatString = pieces.second.drop_front();
    if (next >= '0' && next <= '9') {
      const unsigned index = next - '0';
      if (index < args.size()) {
        OS << args[index];
      } else {
        OS << "<Argument_Not_Provided>";
      }
    }
  }
}

static void maybeAddLineAndColumn(SourceRange range, llvm::raw_ostream &OS) {
  if (range.start.line > 0 && range.start.column > 0) {
    OS << range.start.line << ":" << range.start.column << ": ";
  }
}

static void printContextFrameToStream(const Diagnostics::ContextFrame &frame,
                                      llvm::raw_ostream &OS) {
  maybeAddLineAndColumn(frame.range, OS);
  formatErrorString(contextTypeToFormatString(frame.type), frame.args, OS);
}

static void
printMessageToStream(const Diagnostics::ErrorContent::Message &message,
                     const llvm::Twine Prefix, llvm::raw_ostream &OS) {
  maybeAddLineAndColumn(message.range, OS);
  OS << Prefix;
  formatErrorString(errorTypeToFormatString(message.type), message.args, OS);
}

static void printErrorContentToStream(const Diagnostics::ErrorContent &content,
                                      llvm::raw_ostream &OS) {
  if (content.messages.size() == 1) {
    printMessageToStream(content.messages[0], "", OS);
  } else {
    for (size_t i = 0, e = content.messages.size(); i != e; ++i) {
      if (i != 0)
        OS << "\n";
      printMessageToStream(content.messages[i],
                           "Candidate " + llvm::Twine(i + 1) + ": ", OS);
    }
  }
}

void Diagnostics::print(llvm::raw_ostream &OS) const {
  for (const ErrorContent &error : errorValues) {
    if (&error != &errorValues.front())
      OS << "\n";
    printErrorContentToStream(error, OS);
  }
}

void Diagnostics::printFull(llvm::raw_ostream &OS) const {
  for (const ErrorContent &error : errorValues) {
    if (&error != &errorValues.front())
      OS << "\n";
    for (const ContextFrame &frame : error.contextStack) {
      printContextFrameToStream(frame, OS);
      OS << "\n";
    }
    printErrorContentToStream(error, OS);
  }
}

} // namespace mlir::query::matcher
