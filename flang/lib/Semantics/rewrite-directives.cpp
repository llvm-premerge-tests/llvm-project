//===-- lib/Semantics/rewrite-directives.cpp ------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "rewrite-directives.h"
#include "flang/Parser/parse-tree-visitor.h"
#include "flang/Parser/parse-tree.h"
#include "flang/Semantics/semantics.h"
#include "flang/Semantics/symbol.h"
#include <list>
#include <llvm/Frontend/OpenMP/OMP.h.inc>

namespace Fortran::semantics {

using namespace parser::literals;

class DirectiveRewriteMutator {
public:
  explicit DirectiveRewriteMutator(SemanticsContext &context)
      : context_{context} {}

  // Default action for a parse tree node is to visit children.
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

protected:
  SemanticsContext &context_;
};

// Rewrite atomic constructs to add an explicit memory ordering to all that do
// not specify it, honoring in this way the `atomic_default_mem_order` clause of
// the REQUIRES directive.
class OmpRewriteMutator : public DirectiveRewriteMutator {
public:
  explicit OmpRewriteMutator(SemanticsContext &context)
      : DirectiveRewriteMutator(context) {}

  template <typename T> void Walk(T &x) { parser::Walk(x, *this); }
  template <typename T> bool Pre(T &) { return true; }
  template <typename T> void Post(T &) {}

  bool Pre(parser::OpenMPAtomicConstruct &);
  bool Pre(parser::OpenMPRequiresConstruct &);
  void Post(parser::OmpAtomicDefaultMemOrderClause &);

private:
  parser::CharBlock requiresClauseSource_{nullptr};
  bool atomicDirectiveDefaultOrderFound_{false};
};

bool OmpRewriteMutator::Pre(parser::OpenMPAtomicConstruct &x) {
  // Find top-level parent of the operation.
  Symbol *topLevelParent{common::visit(
      [&](auto &atomic) {
        Symbol *symbol{nullptr};
        Scope *scope{
            &context_.FindScope(std::get<parser::Verbatim>(atomic.t).source)};
        do {
          if (Symbol * parent{scope->symbol()}) {
            symbol = parent;
          }
          scope = &scope->parent();
        } while (!scope->IsGlobal());

        assert(symbol &&
            "Atomic construct must be within a scope associated with a symbol");
        return symbol;
      },
      x.u)};

  // Get the `atomic_default_mem_order` clause from the top-level parent.
  std::optional<parser::OmpAtomicDefaultMemOrderClause::Type> defaultMemOrder;
  common::visit(
      [&](auto &details) {
        if constexpr (std::is_convertible_v<decltype(&details),
                          WithOmpDeclarative *>) {
          if (details.has_ompAtomicDefaultMemOrder()) {
            defaultMemOrder = *details.ompAtomicDefaultMemOrder();
          }
        }
      },
      topLevelParent->details());

  if (!defaultMemOrder) {
    return false;
  }

  auto findMemOrderClause =
      [](const std::list<parser::OmpAtomicClause> &clauses) {
        return std::find_if(
                   clauses.begin(), clauses.end(), [](const auto &clause) {
                     return std::get_if<parser::OmpMemoryOrderClause>(
                         &clause.u);
                   }) != clauses.end();
      };

  // Get the clause list to which the new memory order clause must be added,
  // only if there are no other memory order clauses present for this atomic
  // directive.
  std::list<parser::OmpAtomicClause> *clauseList = common::visit(
      common::visitors{[&](parser::OmpAtomic &atomicConstruct) {
                         // OmpAtomic only has a single list of clauses.
                         auto &clauses{std::get<parser::OmpAtomicClauseList>(
                             atomicConstruct.t)};
                         return !findMemOrderClause(clauses.v) ? &clauses.v
                                                               : nullptr;
                       },
          [&](auto &atomicConstruct) {
            // All other atomic constructs have two lists of clauses.
            auto &clausesLhs{std::get<0>(atomicConstruct.t)};
            auto &clausesRhs{std::get<2>(atomicConstruct.t)};
            return !findMemOrderClause(clausesLhs.v) &&
                    !findMemOrderClause(clausesRhs.v)
                ? &clausesRhs.v
                : nullptr;
          }},
      x.u);

  // Add a memory order clause to the atomic directive.
  if (clauseList) {
    atomicDirectiveDefaultOrderFound_ = true;
    switch (*defaultMemOrder) {
    case parser::OmpAtomicDefaultMemOrderClause::Type::AcqRel:
      clauseList->push_back(parser::OmpMemoryOrderClause(
          parser::OmpClause(parser::OmpClause::AcqRel())));
      break;
    case parser::OmpAtomicDefaultMemOrderClause::Type::Relaxed:
      clauseList->push_back(parser::OmpMemoryOrderClause(
          parser::OmpClause(parser::OmpClause::Relaxed())));
      break;
    case parser::OmpAtomicDefaultMemOrderClause::Type::SeqCst:
      clauseList->push_back(parser::OmpMemoryOrderClause(
          parser::OmpClause(parser::OmpClause::SeqCst())));
      break;
    }
  }

  return false;
}

bool OmpRewriteMutator::Pre(parser::OpenMPRequiresConstruct &x) {
  for (parser::OmpClause &clause : std::get<parser::OmpClauseList>(x.t).v) {
    // Store source in case we need to emit an error
    requiresClauseSource_ = clause.source;
    Walk(clause.u);
    requiresClauseSource_ = nullptr;
  }
  return false;
}

void OmpRewriteMutator::Post(parser::OmpAtomicDefaultMemOrderClause &x) {
  if (atomicDirectiveDefaultOrderFound_) {
    context_.Say(requiresClauseSource_,
        "REQUIRES directive with '%s' clause found lexically after atomic "
        "operation without a memory order clause"_err_en_US,
        parser::ToUpperCaseLetters(llvm::omp::getOpenMPClauseName(
            llvm::omp::OMPC_atomic_default_mem_order)
                                       .str()));
  }
}

bool RewriteOmpParts(SemanticsContext &context, parser::Program &program) {
  if (!context.IsEnabled(common::LanguageFeature::OpenMP)) {
    return true;
  }
  OmpRewriteMutator{context}.Walk(program);
  return !context.AnyFatalError();
}

} // namespace Fortran::semantics
